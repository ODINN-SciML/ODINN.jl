using Pkg

# Change back to the `docs` directory if necessary
if basename(pwd()) != "docs"
    cd("docs")
end

Pkg.activate(".")

# Add the local version of the package
Pkg.develop(PackageSpec(path=".."))
Pkg.instantiate()

using Revise
using Documenter, Literate
using ODINN
include("src/doc_utils.jl")

DocMeta.setdocmeta!(ODINN, :DocTestSetup, :(using ODINN); recursive=true)

# Convert tutorial/examples to markdown
Literate.markdown("./src/tutorials.jl", "./src";
                  name = "tutorials", preprocess = replace_includes)

# Which markdown files to compile to HTML
makedocs(
    modules=[ODINN, Huginn, Muninn, Sleipnir],
    authors="Jordi Bolibar, Facu Sapienza",
    repo="https://github.com/ODINN-SciML/ODINN.jl/blob/{commit}{path}#{line}",
    sitename="ODINN.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ODINN-SciML.github.io/ODINN.jl",
        assets=String[],
        size_threshold=500 * 1024,  # Increase size threshold to 500 KiB
        size_threshold_warn=250 * 1024  # Increase warning threshold to 250 KiB
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => "tutorials.md",
        "Types and functions" => "funcs_types.md",
        "API" => "api.md"
    ],
    checkdocs=:none
)