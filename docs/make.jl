using ODINN
using Documenter

DocMeta.setdocmeta!(ODINN, :DocTestSetup, :(using ODINN); recursive=true)

makedocs(;
    modules=[ODINN],
    authors="Jordi Bolibar, Facu Sapienza",
    repo="https://github.com/JordiBolibar/ODINN.jl/blob/{commit}{path}#{line}",
    sitename="ODINN.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JordiBolibar.github.io/ODINN.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JordiBolibar/ODINN.jl",
    devbranch="main",
)
