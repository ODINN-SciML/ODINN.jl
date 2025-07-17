# Differentiability 

In this section we aim to discuss the main choices and strategies regarding differentiability within the ODINN ecosystem. 

We have two main strategies regarding differentiability of hybrid models combining differential equations (e.g. SIA2D) and regressors: manual adjoints and those computed using automatic differentiation (i.e. `Enzyme.jl`).

## Manual adjoints


## AD adjoints

In order to ensure end-to-end differentiability of the whole model using `Enzyme.jl`, special attention needs to be taken in terms of code style and type stability. For this, we leverage the adjoint capabilities of `SciMLSensitivity.jl` from the SciML ecosystem. Here, we compile the main considerations and things that need to be taken into account:

- Ensure and test type stability. All new functions and types need to be type stable. This is crucial for them to be differentiable with `Enzyme.jl`. This is mainly implemented in tests using `JET.@test_opt`, a macro from the JET.jl package that performs static analysis on Julia code to detect potential type instabilities, method errors, or other issues at compile time.