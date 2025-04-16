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

# List of tutorial files
tutorial_files = [
    "src/tutorials/forward_simulation.jl"
    # "src/tutorials/functional_inversion.jl"
]

# Generate independent Markdown files for each tutorial
for tutorial_file in tutorial_files
    tutorial_name = splitext(basename(tutorial_file))[1]  # Extract the file name without extension
    Literate.markdown(tutorial_file, "./src/tutorials"; name = tutorial_name)
end

# Merge tutorials into a single Markdown file
open("./src/tutorials.md", "w") do io
    println(io, "# Tutorials\n")
    for tutorial_file in tutorial_files
        # Extract the file name without extension
        tutorial_name = splitext(basename(tutorial_file))[1]
        
        # Read the content of the generated Markdown file and append it
        tutorial_md_file = joinpath("./src/tutorials", tutorial_name * ".md")
        if isfile(tutorial_md_file)
            tutorial_content = read(tutorial_md_file, String)
            println(io, tutorial_content)
        else
            println(io, "Error: Could not find tutorial file $tutorial_md_file")
        end
    end
end

# Convert tutorial/examples to markdown
# Literate.markdown("./src/tutorials.jl", "./src";
#                   name = "tutorials", preprocess = replace_includes)
# Literate.markdown("./src/tutorials.jl", "./src"; name = "tutorials")

# Which markdown files to compile to HTML
makedocs(
    modules=[ODINN, Huginn, Muninn, Sleipnir],
    authors="Jordi Bolibar, Facu Sapienza, Alban Gossard",
    repo="https://github.com/ODINN-SciML/ODINN.jl/blob/{commit}{path}#{line}",
    sitename="ODINN.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ODINN-SciML.github.io/ODINN.jl",
        assets=String[]
        # size_threshold=500 * 1024,  # Increase size threshold to 500 KiB
        # size_threshold_warn=250 * 1024  # Increase warning threshold to 250 KiB
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => "tutorials.md",
        "Types and functions" => "funcs_types.md",
        "API" => "api.md"
    ],
    checkdocs=:none
)