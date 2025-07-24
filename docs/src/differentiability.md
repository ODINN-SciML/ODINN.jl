# Differentiability 

In this section we aim to discuss the main choices and strategies regarding differentiability within the ODINN ecosystem. 

We have two main strategies regarding differentiability of hybrid models combining differential equations (e.g. SIA2D) and regressors: manual adjoints and those computed using automatic differentiation (i.e. `Enzyme.jl`).

## Manual adjoints


## AD adjoints

In order to ensure end-to-end differentiability of the whole model using `Enzyme.jl`, special attention needs to be taken in terms of code style and type stability. For this, we leverage the adjoint capabilities of `SciMLSensitivity.jl` from the SciML ecosystem. Here, we compile the main considerations and things that need to be taken into account:

- Ensure and test type stability. All new functions and types need to be type stable. This is crucial for them to be differentiable with [`Enzyme.jl`](https://enzymead.github.io/Enzyme.jl/stable/). This is mainly implemented in tests using `JET.@test_opt`, a macro from the [`JET.jl`](https://aviatesk.github.io/JET.jl/stable/) package that performs static analysis on Julia code to detect potential type instabilities, method errors, or other issues at compile time. In particular, structures with abstract fields must use parametric types.
- Make sure that the input variables of the ice flow equation (e.g. `SIA2D!`) are not mutated. As calls to `Enzyme.jl` in `SciMLSensitivity.jl` use [`Enzyme.Const`](https://enzymead.github.io/Enzyme.jl/stable/api/#EnzymeCore.Const) for the input variable `H` when computing the VJP with respect to the parameters, it expects `H` to not be modified in-place. A typical example is `SIA2D!` where in the first lines, `H` is copied even though this is an in-place implementation because we need to clip the negative values of the input variable `H`. Failing to respect this constraint will result in the following error:

!!! danger "ERROR"
    ```
    Constant memory is stored (or returned) to a differentiable variable.
    As a result, Enzyme cannot provably ensure correctness and throws this error.
    This might be due to the use of a constant variable as temporary storage for active memory (https://enzyme.mit.edu/julia/stable/faq/#Runtime-Activity).
    If Enzyme should be able to prove this use non-differentable, open an issue!
    To work around this issue, either:
        a) rewrite this variable to not be conditionally active (fastest, but requires a code change), or
        b) set the Enzyme mode to turn on runtime activity (e.g. autodiff(set_runtime_activity(Reverse), ...) ). This will maintain correctness, but may slightly reduce performance.
    ```

- The [`ODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/) needs to be defined out of the function that is being differentiated. We use `remake(iceflow_prob; p=container)` to define a new `ODEProblem` and the solver will use the parameters defined in `container`.
- No closures are used to provide the `Simulation` object to the iceflow equation. The [SciMLStructures.jl](https://github.com/SciML/SciMLStructures.jl) package has been especially implemented for this use case where one wants to provide both a vector of parameters to optimize, and a complex struct. This struct is designed to store some intermediate results through a cache and simulation parameters that are used to store physical quantities. The documentation provides [an example of how to implement the interface](https://sciml.github.io/SciMLStructures.jl/stable/example/). Special attention should be given to the definition of the `replace` function which should deepcopy the whole struct and zero the fields that are not used to differentiate parameters.
- At the loss function level, all the operations need to be out-of-place as [`Zygote.jl`](https://fluxml.ai/Zygote.jl/stable/) is used to differentiate this part of the computational graph. This means for example that one cannot affect the `results` in `simulation` and this has to be done outside of the functions that are called by `Zygote.gradient`. The error raised in case an in-place affectation is done is rather explicit.
- Parts of the computational graph are not needed to compute the true gradient and they can be bypassed thanks to the `@ignore_derivatives` macro. This is the case for example of the reference ice thickness.
