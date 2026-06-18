using Pkg

# Change back to the `docs` directory if necessary
if basename(pwd()) != "docs"
    cd("docs")
end

Pkg.activate(".")

# Add the local version of the package
Pkg.develop(PackageSpec(path = ".."))
Pkg.instantiate()

ODINN_OVERWRITE_MULTI = get(ENV, "CI", nothing)=="true"
ENV["ODINN_OVERWRITE_MULTI"] = ODINN_OVERWRITE_MULTI
@show ODINN_OVERWRITE_MULTI

# Activate to avoid GKS plots popping up in the REPL in local
ENV["GKSwstype"]="nul"

# # Disable the Blink hack when the target output is for notebooks
# ENV["ODINN_PLOTLYJS_NB"] = "true"

using Revise
using Documenter, Literate
using ODINN
using DocumenterCitations

cd(dirname(Base.active_project()))

bib = CitationBibliography(
    joinpath(@__DIR__, "src/assets", "references.bib");
    style = :numeric
)

DocMeta.setdocmeta!(ODINN, :DocTestSetup, :(using ODINN); recursive = true)

# List of tutorial files
tutorial_files = [
    "./src/forward_simulation.jl",
    "./src/classical_inversion.jl",
    "./src/functional_inversion.jl",
    "./src/laws.jl",
    "./src/vjp_laws.jl",
    "./src/input_laws.jl",
    "./src/quick_start.jl",
    "./src/results_plotting_tutorial.jl"
]

# Generate independent Markdown files for each tutorial.
# Set ODINN_SKIP_LITERATE=true to reuse previously generated .md files (faster local iteration).
if get(ENV, "ODINN_SKIP_LITERATE", "false") != "true"
    for tutorial_file in tutorial_files
        tutorial_name = splitext(basename(tutorial_file))[1]  # Extract the file name without extension
        Literate.markdown(tutorial_file, "./src"; name = tutorial_name)
    end
end

# Which markdown files to compile to HTML
makedocs(
    modules = [ODINN, Huginn, Muninn, Sleipnir],
    authors = "Jordi Bolibar, Facu Sapienza, Alban Gossard, Mathieu le Séac'h, Vivek Gajadhar",
    repo = Remotes.GitHub("ODINN-SciML", "ODINN.jl"),
    sitename = "ODINN.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing)=="true",
        ansicolor = true, collapselevel = 3,
        size_threshold = 2000 * 1024,  # Increase size threshold to 500 KiB
        size_threshold_warn = 1000 * 1024,  # Increase warning threshold to 250 KiB),      # in bytes
        example_size_threshold = 1000 * 1024
    ),
    pages = [
        "Home" => "index.md",
        "Quick start" => "quick_start.md",
        "Ecosystem packages" => [
            "Sleipnir.jl" => "Packages/sleipnir.md",
            "Muninn.jl" => "Packages/muninn.md",
            "Huginn.jl" => "Packages/huginn.md",
            "Gungnir" => "Packages/gungnir.md",
            "ODINN.jl" => "Packages/odinn.md"
        ],
        "How to use ODINN" => [
            "Parameters" => "parameters.md",
            "Glaciers" => "glaciers.md",
            "Models" => "models.md",
            "Results and plotting" => "results_plotting.md",
            "Plotting tutorial" => "results_plotting_tutorial.md"
        ],
        "Inversions" => [
            "Inversion types" => "inversions.md",
            "Optimization" => "optimization.md",
            "Sensitivity analysis" => "sensitivity.md"
        ],
        "Tutorials" => [
            "Forward simulation" => "forward_simulation.md",
            "Classical inversion" => "classical_inversion.md",
            "Functional inversion" => "functional_inversion.md",
            "Laws" => "laws.md",
            "Laws inputs" => "input_laws.md",
            "Laws VJP customization" => "vjp_laws.md"
        ],
        "API" => [
            "Sleipnir.jl" => "API/api_sleipnir.md",
            "Muninn.jl" => "API/api_muninn.md",
            "Huginn.jl" => "API/api_huginn.md",
            "ODINN.jl" => "API/api_odinn.md"
        ],
        "Community" => [
            "How to contribute" => "contribute.md",
            "Code of conduct" => "code_of_conduct.md"
        ], "Ongoing changes and future plans" => "changes_plans.md",
        "References" => "references.md"
    ],
    checkdocs = :none,
    plugins = [bib]
)

if get(ENV, "CI", nothing)=="true"
    deploydocs(
        repo = "github.com/ODINN-SciML/ODINN.jl",
        branch = "gh-pages",
        devbranch = "main",
        push_preview = true,
        forcepush = true
    )
end
