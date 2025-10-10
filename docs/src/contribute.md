# How to contribute

We welcome all types of contributions to the ODINN ecosystem! Most of the code, documentation, and features of ODINN are constantly under development, so your feedback can be very helpful! If you are interested in contributing, there are many ways in which you can help:

- **Report bugs in the code.** You can report problems with the code by opening issues under the [issues](https://github.com/ODINN-SciML/ODINN.jl/issues) tab in the ODINN repository. Please explain the problem you encounter and try to give a complete description of it so we can follow up on that.
- **Request new features and documentation.** If there is an important topic or example that you feel falls under the scope of this project and you would like us to include it, please request it! We are looking for new insights into what the community wants to learn.
- **Contribute to the source code.** We welcome pull requests (PRs) to any  of the libraries in the ODINN ecosystem. In order to contribute, please make a fork of the repository you would like to contribute to, and then submit a PR to:
  - the `dev` branch in [ODINN.jl](https://github.com/ODINN-SciML/ODINN.jl/) (`main` is only updated from time to time when enough meaningful commits are available to perform a new release);
  - the `main` branch in [Sleipnir.jl](https://github.com/ODINN-SciML/Sleipnir.jl), [Muninn.jl](https://github.com/ODINN-SciML/Muninn.jl) and [Huginn.jl](https://github.com/ODINN-SciML/Huginn.jl/).

We will review your PR it and provide feedback. If you are looking for ideas of how to contribute with code, you can check the [opened issues](https://github.com/ODINN-SciML/ODINN.jl/issues) in our repositories.

!!! tip
    If you need help navigating the world of PRs and contributing in GitHub, we encourage you to take a look at the [tutorial](https://docs.oggm.org/en/stable/contributing.html) put together by our OGGM friends.

## Coding style

We use [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl) to ensure consistent coding style.
If you open a PR, your changes must follow the SciML coding style.
One way to ensure that each of your commits always follow the SciML style is to install a commit hook in the packages you are developing.

First you need to install [pre-commit](https://pre-commit.com/) which allows you to easily install a commit hook:
```bash
pip install pre-commit
```
Installing the commit hook defined in the `.pre-commit-config.yaml` file at the root of the repository can be done by running:
```bash
pre-commit install
```
Then once you have staged your changes, when running the `git commit` command, the hook will trigger and `JuliaFormatter` will ask you to confirm the formatting that have been applied (if changes to the code format were necessary).

!!! tip
    You need to install this commit hook in each of the repositories you are editing.

!!! tip "Continous integration"
    A github action checks within the CI that each commit follows the SciML style and you will get an error if by any chance you pushed changes that need to be formatted.

## Contributing to the documentation

Here we show the basics around building the docs locally and making contributions.

!!! warning "Multiprocessing in the documentation"

    In order to use multiprocessing in the documentation, we set up a specific number of workers in the Julia session in the [`documentation.yml`](https://github.com/ODINN-SciML/ODINN.jl/blob/main/.github/workflows/documentation.yml) file. It is imperative that the number of workers set there matches the ones set in the Julia code run in the documentation. By default, we have set them to `-p 3` in `documentation.yml`, meaning that 3 workers will be added on top of the head one. This will match the default number of workers in `SimulationParameters`, but if you manually specify them, make sure to set them to 4 (the number of parameters in ODINN DOES include the head worker). This is often a source of confusion, so refrain from playing with the number of workers in the documentation. 

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

This will print a localhost URL that you can open in your browser to visualize the documentation.