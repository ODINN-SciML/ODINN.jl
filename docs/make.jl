using Pkg

# Change back to the `docs` directory if necessary
if basename(pwd()) != "docs"
    cd("docs")
end

Pkg.activate(".")

# Add the local version of the package
Pkg.develop(PackageSpec(path=".."))
Pkg.instantiate()

ODINN_OVERWRITE_MULTI = get(ENV, "CI", nothing)=="true"
ENV["ODINN_OVERWRITE_MULTI"] = ODINN_OVERWRITE_MULTI
@show ODINN_OVERWRITE_MULTI

using Revise
using Documenter, Literate
using ODINN
using DocumenterCitations

bib = CitationBibliography(
    joinpath(@__DIR__, "src/assets", "references.bib");
    style=:numeric
)

DocMeta.setdocmeta!(ODINN, :DocTestSetup, :(using ODINN); recursive=true)

# List of tutorial files
tutorial_files = [
    "./src/forward_simulation.jl",
    "./src/functional_inversion.jl",
    "./src/laws.jl",
    "./src/quick_start.jl"
]

# Generate independent Markdown files for each tutorial
for tutorial_file in tutorial_files
    tutorial_name = splitext(basename(tutorial_file))[1]  # Extract the file name without extension
    Literate.markdown(tutorial_file, "./src"; name = tutorial_name)
end

# Which markdown files to compile to HTML
makedocs(
    modules=[ODINN, Huginn, Muninn, Sleipnir],
    authors="Jordi Bolibar, Facu Sapienza, Alban Gossard, Mathieu le Séac'h, Vivek Gajadhar",
    repo=Remotes.GitHub("ODINN-SciML", "ODINN.jl"),
    sitename="ODINN.jl",
    format = Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing)=="true",
        ansicolor=true, collapselevel=3,
        size_threshold=2000 * 1024,  # Increase size threshold to 500 KiB
        size_threshold_warn=1000 * 1024,  # Increase warning threshold to 250 KiB),      # in bytes
        example_size_threshold=1000 * 1024
    ),
    pages=[
        "Home" => "index.md",
        "Quick start" => "quick_start.md",
        "Tutorials" => [
            "Forward simulation" => "forward_simulation.md",
            "Functional inversion" => "functional_inversion.md",
            "Laws" => "laws.md",
        ],
        "How to use ODINN" => [
        "Parameters" => "parameters.md",
        "Glaciers" => "glaciers.md",
        "Models" => "models.md",
        "Results and plotting" => "results_plotting.md",
        "API" => "api.md",
        ],
        "Differentiability" => "differentiability.md",
        "Code style and recommendations" => "style_recommendations.md",
        "Ongoing changes and future plans" => "changes_plans.md",
        "References" => "references.md",
    ],
    checkdocs=:none,
    plugins=[bib]
)

if get(ENV, "CI", nothing)=="true"
    deploydocs(
        repo = "github.com/ODINN-SciML/ODINN.jl",
        branch = "gh-pages",
        devbranch = "main",
        push_preview = true,
        forcepush = true,
    )
end
