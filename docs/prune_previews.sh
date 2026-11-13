#!/usr/bin/env bash
# Remove the previews of closed (merged or not) pull requests from the gh-pages branch.
#
# Documenter builds a preview of the docs in previews/PR<N> for every pull request, and
# nothing removes it afterwards. Each one is a full copy of the docs (~10 MB), and deploying
# pushes the whole branch, so the stale ones make the push fail (HTTP/2 error on the
# gh-pages push) once the branch gets too large.
#
# Only previews/PR<N> folders are ever touched, and only if GitHub says PR <N> is closed.
# If anything is unclear (API error, not a PR, unexpected change) the folder is kept.
#
# The new commit is built with git plumbing from the trees of the branch only, so the
# hundreds of MB of files in the previews are never downloaded.
#
# Environment variables:
#   REPO        owner/name of the repository (required)
#   GH_TOKEN    token that can read pull requests and push to gh-pages (required)
#   CURRENT_PR  number of the pull request being built, never removed (optional)
#   DRY_RUN     set to 1 to list and check everything without pushing anything
set -euo pipefail

REPO="${REPO:?REPO (owner/name) is required}"
: "${GH_TOKEN:?GH_TOKEN is required}"
BRANCH="${BRANCH:-gh-pages}"
# Where the result is pushed, only different from BRANCH to test the script without touching it
PUSH_BRANCH="${PUSH_BRANCH:-$BRANCH}"
BACKUP_BRANCH="${BACKUP_BRANCH:-gh-pages-backup}"
CURRENT_PR="${CURRENT_PR:-}"
DRY_RUN="${DRY_RUN:-0}"

# Without a working API there is no way to know which pull requests are closed
if ! gh api "repos/${REPO}" --jq .full_name > /dev/null; then
    echo "::warning::Could not read ${REPO} from the GitHub API, nothing was removed."
    exit 0
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

# The token goes through the environment, never through a command line or a URL
auth="$(printf 'x-access-token:%s' "$GH_TOKEN" | base64 | tr -d '\n')"
export GIT_TERMINAL_PROMPT=0
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="http.https://github.com/.extraheader"
export GIT_CONFIG_VALUE_0="AUTHORIZATION: basic ${auth}"

# Only the trees of the branch are needed, not the files in them
git clone --quiet --depth 1 --filter=blob:none --no-checkout --branch "$BRANCH" \
    "https://github.com/${REPO}.git" "$workdir/repo"
cd "$workdir/repo"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

tip="$(git rev-parse HEAD)"

folders=()
while IFS= read -r folder; do
    folders+=("$folder")
done < <(git ls-tree --name-only "${tip}:previews" | { grep -E '^PR[0-9]+$' || true; })
echo "Found ${#folders[@]} previews on ${BRANCH}."

remove_list="$workdir/remove.txt"
: > "$remove_list"
for folder in ${folders[@]+"${folders[@]}"}; do
    number="${folder#PR}"
    if [ "$number" = "$CURRENT_PR" ]; then
        echo "keep   ${folder} (being built now)"
        continue
    fi
    state="$(gh api "repos/${REPO}/pulls/${number}" --jq .state 2> /dev/null || echo unknown)"
    case "$state" in
        closed)
            echo "remove ${folder} (closed)"
            echo "$folder" >> "$remove_list"
            ;;
        open)
            echo "keep   ${folder} (open)"
            ;;
        *)
            echo "keep   ${folder} (state: ${state})"
            ;;
    esac
done

n_remove="$(wc -l < "$remove_list" | tr -d ' ')"
if [ "$n_remove" -eq 0 ]; then
    echo "No preview of a closed pull request to remove."
    exit 0
fi

# previews/ without the closed ones, then the root with that previews/, then a commit
new_previews="$(git ls-tree "${tip}:previews" \
    | awk -F'\t' 'NR == FNR { drop[$0] = 1; next } !($2 in drop)' "$remove_list" - \
    | git mktree --missing)"
new_root="$(git ls-tree "$tip" \
    | awk -F'\t' -v p="$new_previews" '{
        split($1, a, " ")
        if ($2 == "previews") { print a[1] " " a[2] " " p "\t" $2 } else { print }
    }' \
    | git mktree --missing)"
new_commit="$(git commit-tree "$new_root" -p "$tip" \
    -m "Remove the previews of closed pull requests" \
    -m "Removed ${n_remove} folders of previews/, by docs/prune_previews.sh.")"

# Only deletions inside previews/PR<N>/ are allowed in this commit
unexpected="$(git diff-tree -r --name-status "$tip" "$new_commit" \
    | awk '$1 != "D" || $2 !~ /^previews\/PR[0-9]+\//' | head -n 1)"
if [ -n "$unexpected" ]; then
    echo "::error::Unexpected change in the commit, aborting: ${unexpected}"
    exit 1
fi

# The published site must still be there
n_site="$(git ls-tree "$new_commit" -- dev index.html versions.js | wc -l | tr -d ' ')"
if [ "$n_site" -ne 3 ]; then
    echo "::error::dev, index.html or versions.js is missing from the commit, aborting."
    exit 1
fi

# What a push would send: only the new trees, as it should be (informative, never fatal)
if pack_bytes="$(printf '%s\n^%s\n' "$new_commit" "$tip" \
    | git pack-objects --revs --thin --stdout 2> /dev/null | wc -c | tr -d ' ')"; then
    echo "The push would send ${pack_bytes} bytes."
else
    echo "Could not measure the size of the push."
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry run] ${n_remove} previews would be removed, ${BACKUP_BRANCH} would be created if missing."
    git push --dry-run origin "${new_commit}:refs/heads/${PUSH_BRANCH}"
    exit 0
fi

# Keep a copy of the branch as it was, once. This only adds a ref, no new objects.
if ! git ls-remote --exit-code --heads origin "$BACKUP_BRANCH" > /dev/null 2>&1; then
    git push --quiet origin "${tip}:refs/heads/${BACKUP_BRANCH}"
    echo "Created ${BACKUP_BRANCH}."
fi

# No force: if another job updated the branch in the meantime, the next run will retry
if git push --quiet origin "${new_commit}:refs/heads/${PUSH_BRANCH}"; then
    echo "Removed ${n_remove} previews."
else
    echo "::warning::The push to ${PUSH_BRANCH} was rejected, nothing was removed this time."
fi
