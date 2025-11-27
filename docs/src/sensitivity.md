# Sensitivity Analysis

In this section we aim to discuss the main choices and strategies regarding sensitivity analysis within the ODINN ecosystem.
Sensitivity analysis is important in order to differentiate the different ice flow simulations and enable parameter calibration during the inverse modelling stage.

ODINN currently supports two main strategies regarding the computation of model sensitivity of hybrid models combining differential equations (e.g. SIA2D) and regressors: manual adjoints and [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/).

## Manual adjoints

ODINN includes an implementation of both discrete and continuous adjoint methods (see [sapienza_differentiable_2024](@cite) for a complete overview of these different methods).
In both cases, the adjoint variable is computed as the solution of a time reversed differential equation.
The adjoint state variable is then used to efficiently compute the gradient of a prescribed loss function.

For the SIA2D equation, the iceflow equations can be written as
```math
    \frac{\partial H}{\partial t}
    =
    \dot b
    +
    \nabla \cdot
    \left(
    D(H, \nabla S)
    \nabla S
    \right)
```
where $D$ is a diffusivity term (see [cuffey_physics_2010](@cite) for more information).
Given a loss function $L(\theta)$ (see the [Optimization](./optimization.md) section), the adjoint variable $\lambda$ is defined as the solution of the following partial differential equation:
```math
    \frac{\partial \lambda}{\partial t}
    =
    - \nabla \cdot \left( D \nabla \lambda \right)
    + \frac{\partial D}{\partial H} \nabla S \cdot \nabla \lambda
    - \nabla \cdot \left( \frac{\partial D}{\partial (\nabla H)} \nabla S \cdot \nabla \lambda \right)
    - \frac{\partial \ell}{\partial H}
```
with final condition $\lambda(x,y,t_1) = 0$ and $\lambda |_{\partial \Omega} \equiv 0$.
The gradient of the loss function $L(\theta)$ with respect to the parameter $\theta$ then can be computed using the following expression:
```math
    \frac{dL}{d\theta_i}
    =
    - \iint
    \frac{\partial D}{\partial \theta_i} \nabla S \cdot \nabla \lambda\, \mathrm{d}t\, \mathrm{d}\Omega
    +
    \iint \frac{\partial \ell}{\partial \theta_i}\, \mathrm{d}t\, \mathrm{d}\Omega
```
The integration of the adjoint equation can be performed using the discrete adjoint (discretize-then-differentiate) or the continuous adjoint (differentiate-then-discretize).
Both types of adjoints are implemented as an `AbstractAdjointMethod`:
- `DiscreteAdjoint`: The discrete adjoint is a very simple adjoint that uses an explicit Euler scheme to solve the adjoint ODE. The timestep is prescribed by the frequency at which the results are saved in the forward run. It is usually set to one month but it can also adapt when the loss function is evaluated at non uniform time steps.
- `ContinuousAdjoint`: With this adjoint method, the adjoint ODE is treated and solved as a standard ODE using [`DifferentialEquations.jl`](https://diffeq.sciml.ai/). The VJP with respect to the ice thickness of the ice flow equation (e.g. `SIA2D!`) is integrated backward in time. The gradients with respect to the parameters involved in the iceflow equation are then computed using a simple Gauss Quadrature at prescribed time steps. These time steps are determined by the Gauss Quadrature method and in a general case they are different from the time steps at which results are gathered in the forward run. This way the adjoint solution is interpolated. For the moment only linear interpolators are supported.

The default choice for the manual adjoint is `ContinuousAdjoint`, which relies on [`DifferentialEquations.jl`](https://diffeq.sciml.ai/) for solving the reverse adjoint equations, then
providing better error control on the computation of the gradients.

### Computing the VJPs inside the solver

When evaluating the adjoint differential equations used to compute the gradient of the loss function, vector-Jacobian products (VJPs) need to be evaluated at every given timestep (see Section 4.2.1.1 in [sapienza_differentiable_2024](@cite)).
These VJPs are then used inside both continuous and discrete adjoints, where the adjoint equation is integrated in time. 
The computation of these VJPs can be efficiently computed using automatic differentiation.
ODINN provides manual implementations of the pullback operations required to compute these VJPs, together with the interface to
compute these VJPs using the native Julia automatic differentiation libraries.
The VJP methods in ODINN are implemented as concrete types of `AbstractVJPMethod`:
- `EnzymeVJP`: The Enzyme VJPs rely on [`Enzyme.jl`](https://enzymead.github.io/Enzyme.jl/) to compute the (spatially) discrete VJPs of the iceflow equation. It corresponds to the true VJP of the numerical code.
- `DiscreteVJP`: This is a manual implementation of what the (spatially) discrete Enzyme VJP does. Equations were derived manually by differentiating the discretized differential operators. For example, this means that the partial derivative $\frac{\partial f}{\partial x}$ is first discretized as, for example, `df[i] = (f[i + 1] - f[i]) / dx` and then the pullback operator is directly applied to the discretization `df`.
- `ContinuousVJP`: In the special case of `SIA2D!`, as we are dealing with a diffusion equation, a (spatially) continuous VJP can be derived by integrating by parts the spatial differential operators inside the SIA equation. This means the pullback operator of the differentiation step $\frac{\partial f}{\partial x}$ is first computed before discretizing. It is then discretized after differentiation


## SciMLSensitivity

Gradients can also be computed using [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/). It enables to automate the computation of the adjoint method and the VJPs (Vector-Jacobian Products), done using automatic differentiation via `Enzyme.jl`.

In order to ensure end-to-end differentiability of the whole model using `Enzyme.jl`, special attention needs to be taken in terms of code style and type stability. For this, we leverage the adjoint capabilities of `SciMLSensitivity.jl` from the SciML ecosystem.

Here, we compile the main considerations and things that need to be taken into account when developing differentiable code within ODINN:

- **Ensure and test type stability**. All new functions and types need to be type stable. This is crucial for them to be differentiable with [`Enzyme.jl`](https://enzymead.github.io/Enzyme.jl/stable/). This is mainly implemented in tests using `JET.@test_opt`, a macro from the [`JET.jl`](https://aviatesk.github.io/JET.jl/stable/) package that performs static analysis on Julia code to detect potential type instabilities, method errors, or other issues at compile time. In particular, structures with abstract fields must use parametric types.
- **Make sure that the input variables of the ice flow equation (e.g. `SIA2D!`) are not mutated**. As calls to `Enzyme.jl` in `SciMLSensitivity.jl` use [`Enzyme.Const`](https://enzymead.github.io/Enzyme.jl/stable/api/#EnzymeCore.Const) for the input variable `H` when computing the VJP with respect to the parameters, it expects `H` to not be modified in-place. A typical example is `SIA2D!` where in the first lines, `H` is copied even though this is an in-place implementation because we need to clip the negative values of the input variable `H`. Failing to respect this constraint will result in the following error:

!!! danger "ERROR"
    ```
    Constant memory is stored (or returned) to a differentiable variable.
    As a result, Enzyme cannot provably ensure correctness and throws this error.
    This might be due to the use of a constant variable as temporary storage for active memory (https://enzyme.mit.edu/julia/stable/faq/#Runtime-Activity).
    If Enzyme should be able to prove this use non-differentiable, open an issue!
    To work around this issue, either:
        a) rewrite this variable to not be conditionally active (fastest, but requires a code change), or
        b) set the Enzyme mode to turn on runtime activity (e.g. autodiff(set_runtime_activity(Reverse), ...) ). This will maintain correctness, but may slightly reduce performance.
    ```

- **The [`ODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/) needs to be defined out of the function that is being differentiated**. We use `remake(iceflow_prob; p=container)` to define a new `ODEProblem` and the solver will use the parameters defined in `container`.
- **No closures are used to provide the `Simulation` object to the iceflow equation**. The [SciMLStructures.jl](https://github.com/SciML/SciMLStructures.jl) package has been especially implemented for this use case where one wants to provide both a vector of parameters to optimize, and a complex struct. This struct is designed to store some intermediate results through a cache and simulation parameters that are used to store physical quantities. The documentation provides [an example of how to implement the interface](https://sciml.github.io/SciMLStructures.jl/stable/example/). Special attention should be given to the definition of the `replace` function which should deepcopy the whole struct and zero the fields that are not used to differentiate parameters.
- **At the loss function level, all the operations need to be out-of-place** as [`Zygote.jl`](https://fluxml.ai/Zygote.jl/stable/) is used to differentiate this part of the computational graph. This means for example that one cannot affect the `results` in `simulation` and this has to be done outside of the functions that are called by `Zygote.gradient`. The error raised in case an in-place affectation is done is rather explicit.
- Parts of the computational graph are not needed to compute the true gradient and they can be bypassed thanks to the `@ignore_derivatives` macro. This is the case for example of the reference ice thickness.
