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
    authors="Jordi Bolibar, Facu Sapienza",
    repo="https://github.com/ODINN-SciML/ODINN.jl/blob/{commit}{path}#{line}",
    sitename="ODINN.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ODINN-SciML.github.io/ODINN.jl",
        assets=String[]
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => "tutorials.md",
        "Index of functions and types" => "funcs_types.md"
    ],
    checkdocs=:exports
)

deploydocs(
    repo = "github.com/ODINN-SciML/ODINN.jl",
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
    forcepush = true,
)
