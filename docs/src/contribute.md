# How to contribute

We welcome all types of contributions to the ODINN ecosystem! Most of the code, documentation, and features of ODINN are constantly under development, so your feedback can be very helpful! If you are interested in contributing, there are many ways in which you can help:

  - **Report bugs in the code.** You can report problems with the code by opening issues under the [issues](https://github.com/ODINN-SciML/ODINN.jl/issues) tab in the ODINN repository. Please explain the problem you encounter and try to give a complete description of it so we can follow up on that.

  - **Request new features and documentation.** If there is an important topic or example that you feel falls under the scope of this project and you would like us to include it, please request it! We are looking for new insights into what the community wants to learn.
  - **Contribute to the source code.** We welcome pull requests (PRs) to any  of the libraries in the ODINN ecosystem. In order to contribute, please make a fork of the repository you would like to contribute to, and then submit a PR to:
    
      + the `dev` branch in [ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl/) (`main` is only updated from time to time when enough meaningful commits are available to perform a new release);
      + the `main` branch in [Sleipnir.jl](https://github.com/ODINN-SciML/Sleipnir.jl), [Muninn.jl](https://github.com/ODINN-SciML/Muninn.jl) and [Huginn.jl](https://github.com/ODINN-SciML/Huginn.jl/).

We will review your PR and provide feedback. If you are looking for ideas of how to contribute with code, you can check the opened issues in our repositories: [Sleipnir.jl issues](https://github.com/ODINN-SciML/Sleipnir.jl/issues), [Muninn.jl issues](https://github.com/ODINN-SciML/Muninn.jl/issues), [Huginn.jl issues](https://github.com/ODINN-SciML/Huginn.jl/issues) and [ODINN.jl issues](https://github.com/ODINN-SciML/ODINN.jl/issues).

!!! tip
    
    If you need help navigating the world of PRs and contributing in GitHub, we encourage you to take a look at the [tutorial](https://docs.oggm.org/en/stable/contributing.html) put together by our OGGM friends.

## SciML code style

In the [ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl/) ecosystem, we follow the [SciML code style](https://github.com/SciML/SciMLStyle).
If you make changes to [Sleipnir.jl](https://github.com/ODINN-SciML/Sleipnir.jl), [Muninn.jl](https://github.com/ODINN-SciML/Muninn.jl), [Huginn.jl](https://github.com/ODINN-SciML/Huginn.jl/) or [ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl/), make sure that you have installed the code formatting tool before committing any change.

!!! warning
    
    If you open a PR with changes that were not properly formatted, the CI will raise an error and in any case a PR that does not follow the coding style cannot be merged.

We use [`JuliaFormatter.jl`](https://github.com/domluna/JuliaFormatter.jl) to format the code.
This can be run automatically everytime you commit changes by using [pre-commit](https://pre-commit.com/) which installs a pre-commit hook.
The pre-commit hook is defined at the root of each repository, for example [here for ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl/blob/main/.pre-commit-config.yaml).

### Install the pre-commit hook

In order to install the pre-commit hook, make sure that you have installed [pre-commit](https://pre-commit.com/) in a Python environment:

```bash
pip install pre-commit
```

Then in the environment where you have installed pre-commit, simply run:

```bash
pre-commit install
```

!!! note
    
    You have to install the pre-commit hook (second command) in each of the packages you edit.

!!! note
    
    When committing changes, you don't need to be in the Python environment where `pre-commit` has been installed. This environment is used only for the installation of the hook.

### Commit changes

Once you have staged your changes, when running the `git commit` command, the hook will trigger and the JuliaFormatter will ask you to confirm the formatting that have been applied (if changes to the code format were necessary).

## Contributing to the documentation

Here we show the basics around building the docs locally and making contributions.

!!! warning "Multiprocessing in the documentation"
    
    In order to use multiprocessing in the documentation, we set up a specific number of workers in the Julia session in the [`documentation.yml`](https://github.com/ODINN-SciML/ODINN.jl/blob/main/.github/workflows/documentation.yml) file. It is imperative that the number of workers set there matches the ones set in the Julia code run in the documentation. By default, we have set them to `-p 3` in `documentation.yml`, meaning that 3 workers will be added on top of the head one. This will match the default number of workers in `SimulationParameters`, but if you manually specify them, make sure to set them to 4 (the number of parameters in ODINN **DOES** include the head worker). This is often a source of confusion, so refrain from playing with the number of workers in the documentation.

### Running the documentation in local

This section contains the instructions to run the documentation locally.

To generate the documentation on your local computer, in the `docs/` folder run:

```julia
include("make.jl")
```

Then in another REPL, in the `docs/` folder, activate the docs environment and run the server:

```julia
using Pkg
Pkg.activate()
using LiveServer
serve()
```

This will print a localhost URL that you can open in your browser. Then, click on the `build` folder to visualize the documentation.

!!! note "What to do when `make.jl` freezes?"
    
    If the building of the documentation freezes, there can be several reasons that cause this. First try to run `include("testdocs.jl")` which will run the tutorial examples. If there is an error during the execution, this will be easier to spot it as [Literate.jl](https://github.com/fredrikekre/Literate.jl) does not always report the error. If after making sure that the code runs smoothly this still freezes, inspect the generated `.md` files (see the list of files at the beginning of `make.jl`) and check that the markdown file was generated properly (code in `@example` sections).
