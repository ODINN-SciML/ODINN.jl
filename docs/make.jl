using Pkg

# Determine if the script is run from the `docs` directory or the root directory
if basename(pwd()) == "docs"
    # Change to the project's root directory
    cd("..")
end

Pkg.activate("docs")

# Add the local version of the package
Pkg.develop(PackageSpec(path="."))
Pkg.instantiate()

using Documenter, Literate
using ODINN

DocMeta.setdocmeta!(ODINN, :DocTestSetup, :(using ODINN); recursive=true)

# Change back to the `docs` directory if necessary
if basename(pwd()) != "docs"
    cd("docs")
end

# Convert tutorial/examples to markdown
Literate.markdown("./src/tutorial.jl", "./src")

# Which markdown files to compile to HTML
makedocs(
    modules=[ODINN],
    authors="Jordi Bolibar, Facu Sapienza",
    repo="https://github.com/ODINN-SciML/ODINN.jl/blob/{commit}{path}#{line}",
    sitename="ODINN.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ODINN-SciML.github.io/ODINN.jl",
        assets=String[],
        repolink="https://github.com/ODINN-SciML/ODINN.jl",
        size_threshold=400000,  # Increase the size threshold to 400 KiB
        size_threshold_warn=300000,  # Set the warning threshold to 300 KiB
    ),
    pages=[
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/ODINN-SciML/ODINN.jl",
    branch = "gh-pages",
    devbranch = "new_API",
    push_preview = true,
    forcepush = true,
)
