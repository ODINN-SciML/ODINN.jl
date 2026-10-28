# Optimization

In this section, we discuss the aspects of the inversions related to parameter or model state optimization and the associated loss functions.

## Loss functions

The loss function that is being optimized $\mathcal{L}$ consists of a data fidelity term and an optional regularization term.
The data fidelity term represents the empirical error between the predicted state of the glacier (given by the simulated forward model) and observations.
The state of the glacier can be characterized by the thickness $H$ and/or the ice surface velocity $V$.
The ground truth for each of these variables is available through different datasets.
In both cases, the objective is to find the value of the model parameters (e.g, the weights of the neural network, the initial state of the glacier), that minimizes the total loss.

We distinguish between the contributions of the loss function that include observations (empirical loss) and regularization losses that penalize solutions that do not comply with some handcrafted prior.
The total loss function is given by

```math
\mathcal{L}(\theta)
=
\lambda_1 \mathcal{L}_1(\theta)
+ \ldots
\lambda_k \mathcal{L}_k(\theta),
```

where each $\mathcal{L}_i$ is a different contribution to the loss (either empirical or regularization) weighted by an hyperparameter $\lambda_i$.
ODINN supports multiobjective loss functions through `MultiLoss` (see [Combining losses](@ref)).
For example, an objective function consisting of one empirical loss function corresponding to differences in ice thickness and a regularization of the ice surface velocity can be defined as follows

```julia
loss = MultiLoss(
    losses = (LossH(), VelocityRegularization()),
    λs = (0.4, 1e-5)
)
```

This `loss` must then be provided to the `empirical_loss_function` argument of `UDEparameters`.

All the terms follow one convention, so that the weights $\lambda_i$ are easy to choose:

  - A data term is the mean over the observed pixels of $((\text{model} - \text{observation}) / σ)^2$, where $σ$ is the uncertainty of the observation. Each observation (a survey, a DEM, a velocity map) counts once.
  - A regularization is the mean over the glacier of $(|D a| / σ)^2$, where $D$ is the gradient or the Laplacian in physical units and $σ$ is the typical size of $D a$ that the prior allows.

So every term is about 1 when the model is as far from the data or the prior as their uncertainty, at any resolution. This follows the convention of [jouvet_inversion_2023](@cite). The weights $\lambda_i$ are then only a relative importance, and they are 1 by default (`MultiLoss(losses = ...)`).
The value of each term is printed at the start and the end of `run!`, and `loss_terms(θ, simulation)` returns it.

A wrong σ only changes the balance between the terms. A bias of the data (for example the penetration of the radar into snow for a DEM) needs a parameter of its own.

In the following sections, we introduce how to define empirical and regularization loss terms, respectively.

### Empirical loss functions

The empirical error can be as simple as the sum of squares of the error between model and observations, but it can also involve more complex cost functions.
The complete description of the different losses are available in their corresponding docstrings (see the [ODINN.jl API](API/api_odinn.md)) but we provide here a brief summary for each of them.

#### Simple loss functions

There are very simple types which are agnostic to the nature of the variables whose error is being computed (that is $H$ or $V$). These are:

  - `L2Sum`: mean of the squared error divided by its uncertainty, $((a - b) / σ)^2$, inside the glacier.
  - `LogSum`: mean of the squared logarithm of the ratio between ice surface velocities, divided by a relative uncertainty (see [morlighem_spatial_2010](@cite) and [morlighem_inversion_2013](@cite)).

#### Time integrated loss functions

These simple loss functions which define very simple operations are used in more complex loss functions:

  - `LossH`: Loss function over the ice thickness only.
  - `LossV`: Loss function over the ice surface velocity only.
  - `LossHV`: Loss function over both the ice thickness and ice surface velocity. It is a shortcut for a `MultiLoss` of `LossH` and `LossV`, see [Combining losses](@ref).

The loss function for the ice thickness $H$ (similar for ice surface velocity $V$) is mathematically defined as:

```math
\mathcal{L}(\theta)
=
\int_{t\in\mathcal{T}} \int_{x\in\Omega} \ell(\hat H(x, t; \theta), \theta) \,\mathrm{d}t \mathrm{d}\Omega
```

where $\Omega\subset\mathbb{R}^2$ defines the spatial domain where we evaluate the loss (usually this corresponds to areas within the glacier that are at least at a given distance from the borders) and $\mathcal{T}$ is the simulation time window.
The term $\ell(\hat H(x, t; \theta), \theta)$ is the point evaluated loss function.
In the case of the $L^2$ loss, we simply have

```math
\ell(\hat H(x, t; \theta), \theta) = \left(\hat H(t, x) - H(t, x)\right)^2.
```

In the formula above, $\hat H$ and $H$ are written as continuous variables, function of both space and time.
In practice, the iceflow equation is solved on a given grid $(x_i)_{i\leq I}$ where each $x_i\in\mathbb{R}^2$.

However, since we have very sparse observations, data is available only at specific points in time and space.
The ground truth ice thickness, generally coming from ground penetrating radar (GPR) field work from the Glathida database [welty_worldwide_2020](@cite), displays a very strong sparsity, with observations concentrated along radar transects.
Glacier ice surface velocity products , e.g. [millan_ice_2022](@cite), [rabatel_satellite-derived_2023](@cite), are notoriously less sparse, but still present many gaps in the grid due to signal-to-noise-ratio issues from the products.
Let $\mathcal{X}=(x_j,t_j)_{j\leq J}$ define the set of points where there are ground truth measurements.
We assume that $\forall j\leq J,\, x_j\in(x_i)_i$, that is the ground truth measurements are aligned with the simulation grid.
For this setting, the empirical error term can be defined as

```math
\sum_{t} \frac{1}{J_t} \sum_{j \leq J_t} \left( \frac{\hat H(t,x_j)-H(t,x_j)}{σ(t, x_j)} \right)^2
```

with $\hat H(t,x_j)$ the predicted ice thickness at the observation time $t$ and on the node of the simulation grid $x_j$, $J_t$ the number of observed pixels at time $t$, and $σ$ the uncertainty of the observation.

The GlaThiDa surveys are lines of cells, so with the default `L2Sum()` (`distance = 3`) the mask would remove all the pixels, because none is 3 pixels away from the border of the observed area. Use `L2Sum(distance = 0)` with them.

##### Uncertainty of the observations

Each data loss divides the misfit by the uncertainty $σ$ of the observation, in the units of the data. By default:

  - `LossH`: the uncertainty of each cell given by the data (`thicknessData.σ`). For GlaThiDa it is the error of the radar (5 m and 3% of the thickness) plus the representation error: the model cell is the mean over the whole cell, while a survey line only crosses it, so the uncertainty grows with the size of the cells (see `Sleipnir.glathida_cell_σ`). If the thickness data has no uncertainty, `LossH` computes one from the observed thickness $H$, cell by cell, $σ(H) = \sqrt{σ_0^2 + (r H)^2}$ with $σ_0$ = 14 m and $r$ = 9% (see `default_σ`). A thin glacier then gets a smaller uncertainty than a thick one, as the differences between independent surveys of the same glacier do.
  - `LossV` and `LossAvgV`: the error of the velocity map, or 10 m/yr if the map has none.
  - `LossDhdt`: the uncertainty of the data, or 0.2 m/yr.
  - `LossS`: the uncertainty of each DEM, plus a part that the products do not give (`σ_sys`, 5 m by default), see [Inversion of the bed](@ref).

A number replaces it, for example `LossH(σ = 20.0)`.

##### Weight of each observation time

By default every observation time counts once (`weighting = :uniform`), so the loss is a plain sum over the observation times. This suits a few surveys or one velocity map.
With `LossH(weighting = :time_span)` (same for `LossV`), each observation time counts as much as the time it covers: half of the time to the previous observation plus half of the time to the next one. The first and the last observations also cover the time to the start and to the end of the simulation.
Use it for dense time series, for example many velocity maps in the same year, so that they do not dominate the loss.
See `observation_weights` for the details.

#### Time aggregated loss functions

There are cost functions which cannot be written as $\int_{t\in\Tau}\int_{x\in\Omega} \ell(...)$ but which are rather of the form $\ell(\int_{t\in\Tau} ...)$.
These functions need special treatment because they cannot be differentiated at every time step during the manual adjoint computation (see note below).
They are defined as a subtype of `TimeAggregatedLoss` and this class of losses include:

  - `LossDhdt`: Loss function for the mean glacier surface elevation change computed between the beginning and the end of a given time window.
  - `LossAvgV`: Loss function for the mean ice surface velocity computed over a given time window.
  - `LossS`: Loss function for the surface elevation, compared with each DEM of the glacier at the date of the DEM, and divided by the uncertainty of the DEM. It is used to invert the bed (`InitialCondition` with `bed = :free`), and only `SciMLSensitivityAdjoint` supports it.

!!! warning "Advanced features"

    If we consider the case of the average ice surface velocity loss function `LossAvgV`, it is mathematically defined as:

    ```math
    \int_{x\in\Omega}\ell\left(\int_{t\in\tau} \hat V(t,x) d\mu_t(t), V(x) \right)d\mu_x(x)
    ```

    Since $\ell$ is non linear, during the manual adjoint computation, we cannot integrate this contribution using a quadrature in the time reversed solve.
    Hence, it has to be differentiated beforehand and this is why `TimeAggregatedLoss` are handled in a different way than the classical loss functions described in the previous sections.
    This allows keeping numerical efficiency for most applications, but when subtype loss functions of `TimeAggregatedLoss` are being used, we precompute the contributions of these specific loss function terms, which is more computationally expensive if comparison to a classical quadrature.

!!! warning "Advanced features"

    The `LossDhdt` loss function could be writen in a similar form as in the remark above:

    ```math
    \ell\left(\int_{t\in\tau}\int_{x\in\Omega} \hat H(t,x) d\mu_t(t)d\mu_x(x), \text{dhdt}(x) \right)
    ```

    with $d\mu_x(t)=\frac{1}{|\Omega|}$ and $d\mu_t(t)=\delta_{t_1}-\delta_{t_0}$ where $t_0$ and $t_1$ are defined in the loss function and represent the time window used to compute the difference in ice thickness.

### Regularization

Regularizations are very common in inverse modelling as they help to constraint the possible solutions of the problem to physical reasonable values.
From a mathematical and computational perspective, regularization losses are just another type of loss that do not include contribution from observations (and then have no _empirical_ contribution to their value).

ODINN currently supports the following type of regularization, although the development of new regularization should be straightforward from the source code API:

  - `InitialThicknessRegularization`: Penalizes large derivatives in the initial condition of the glacier when this is treated as an optimization variable. This builds on top of `TikhonovRegularization`. It applies at the start of the time window of each glacier, or at the time `t₀` that you give (which must be inside the window).
    By default it is the Laplacian of the change `H₀ - glacier.H₀` (`correction = true`), with a `σ` of 0.3 times the roughness of `glacier.H₀` (`σ_relative = 0.3`), so that one value fits any glacier and grid. The shape of `glacier.H₀` is then kept and only the changes made by the inversion are smooth.
    We use it as the default because small scale changes decay in a few weeks once the simulation starts and create a shock in the first time steps, while smooth changes do not, and because in a test on one glacier it gave a lower error against thickness measurements not used in the inversion than a prior on `H₀` itself.
    With `correction = false` the penalty is applied to `H₀` itself, and with `reg = TikhonovRegularization(...)` you choose the regularization and its own `σ`.
  - `VelocityRegularization`: Penalizes large derivatives in the simulated surface velocity of the glacier. This builds on top of `TikhonovRegularization`.
  - `RheologyRegularization` and `SlidingRegularization`: Penalize large derivatives in a gridded `A` or `C` (with the same `tanh` map as the law), at the start of the simulation. The `σ` of their `TikhonovRegularization` is in the units of `A` or `C`.
  - `BedRegularization`: Prior on the whole bed when it is free, following [jouvet_inversion_2023](@cite): its gradient, with a `σ` equal to the roughness of the starting bed by default (`σ_relative = 1`), and an optional convexity term that rewards thicker ice.

To see the `σ` that `InitialThicknessRegularization` or `BedRegularization` uses for a glacier, call `ODINN.prior_σ(prior, glacier)`. `ODINN.field_roughness(a, glacier; operator)` gives the roughness of any field with the definition of `σ_relative`, for example of the inverted bed.

Regularization and empirical losses can be combined together to construct new form of regularizations.

Regularizations rely on simple losses which are agnostic to the nature of the variable that it takes as input.
The following simple losses are implemented, and the regularizations described above are built on top of them:

  - `TikhonovRegularization`: Very common in geophysical inversion. Given a linear operator $D$, this is the mean over the glacier of $(|D(S)| / σ)^2$, where $S$ is some state variable (e.g., the ice thickness or ice surface velocity). The default is the gradient (`operator = :gradient`), as in [jouvet_inversion_2023](@cite). The Laplacian $\nabla^2$ is also available (`operator = :laplacian`), and it is what `InitialThicknessRegularization` uses by default.

### Combining losses

An inversion usually has more than one term: several observations, and often a regularization. Two objects put losses together:

  - `MultiLoss(losses = (...), λs = (...))`: the weighted sum $\sum_i \lambda_i \mathcal{L}_i$ of any number of terms. A term can be any empirical loss (`LossH`, `LossV`, `LossAvgV`, `LossDhdt`, `LossS`) or any regularization. The weights `λs` are 1 by default.
  - `LossHV(hLoss = LossH(), vLoss = LossV(), scaling = 1.0)`: exactly two terms, a thickness loss and a velocity loss, $\mathcal{L}_H + \text{scaling} \cdot \mathcal{L}_V$.

When both can be used they are the same loss, with the same gradient: `LossHV(scaling = s)` is the same as `MultiLoss(losses = (LossH(), LossV()), λs = (1.0, s))`. So `LossHV` is only a shorter way to write this one case. Which one to use depends on what else the inversion needs:

| You want                                                                                                  | Use                                                                          |
|:--------------------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------- |
| Only thickness and velocity, with one factor to balance them                                              | `LossHV` or `MultiLoss`, they are the same                                   |
| A regularization (initial thickness, bed, `A`, `C`, velocity)                                             | `MultiLoss`                                                                  |
| A time aggregated loss (`LossS`, `LossAvgV`, `LossDhdt`)                                                  | `MultiLoss`. A `LossHV` cannot hold it: the `Inversion` stops with an error. |
| A weight for the thickness term too                                                                       | `MultiLoss`                                                                  |
| To see the thickness and the velocity terms separately in `loss_terms` and in the table printed by `run!` | `MultiLoss`. A `LossHV` is shown as one term.                                |

We recommend `MultiLoss` for an inversion, because gridded unknowns almost always need a regularization. `LossHV` suits a quick test or a tutorial that only fits thickness and velocity. They can be mixed, since a `LossHV` is one term of a `MultiLoss`, for example `MultiLoss(losses = (LossHV(), LossS(), BedRegularization()))`, but then `loss_terms` shows the thickness and the velocity as one term.

## Optimizers

The optimizer is set in `Hyperparameters`. It can be one optimizer, or a vector of optimizers used one after the other, with one number of epochs for each of them.
By default, ODINN uses Adam and then LBFGS, for 25 epochs each:

```julia
hyper = Hyperparameters(
    optimizer = [
        ODINN.Adam(0.01),
        ODINN.LBFGS(
            alphaguess = ODINN.LineSearches.InitialStatic(alpha = 1.0, scaled = true),
            linesearch = ODINN.LineSearches.BackTracking(iterations = 5))
    ],
    epochs = [25, 25])
```

Adam is robust far from the minimum, since its step does not depend on the size of the gradient.
LBFGS then converges faster near the minimum.
LBFGS only keeps the last steps in memory, while BFGS stores a matrix of size `n × n` for `n` parameters, which is too large for gridded parameters.
For small problems, like a neural network, `BFGS(initial_stepnorm = 0.001)` can need fewer iterations.

!!! warning "First step of LBFGS"

    At its first iteration, LBFGS has no information about the curvature of the loss, so it tries the step $\theta - \nabla_\theta \mathcal{L}$.
    When the loss is large, this step is large too, and the forward simulation can blow up.
    `InitialStatic(scaled = true)` limits the norm of this trial step to one.

If the forward simulation fails at a point tried by the line search of BFGS or LBFGS, for example because the step was too large, the loss at this point is infinite and the line search takes a shorter step. This works with a line search that only uses the loss, such as `BackTracking` in `default_optimizer()`. A line search that also uses the gradient at the trial point, such as `HagerZhang` (the default of `Optim.BFGS` and `Optim.LBFGS`), still stops with the error.
Adam has no line search, so in this case it stops with the error of the simulation.

## Logging

`ODINN.jl` provides useful statistics, such as the training loss history or the parameters history in the inversion objects.
The training statistics can also be inspected with [TensorBoard](https://www.tensorflow.org/tensorboard) through VSCode or a local webserver.
You can either use the [TensorBoard VSCode extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.tensorboard) or simply install TensorBoard in a Python environment and then launch `tensorboard --logdir .logs/`.

By default the TensorBoard logs are stored in your ODINN folder in `ODINN/.logs/` but you may have to adapt the command above if you log in a different folder.
