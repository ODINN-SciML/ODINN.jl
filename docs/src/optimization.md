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
ODINN supports multiobjective loss functions throught `MultiLoss`.
For example, an objective function consisting of one empirical loss function corresponding to differences in ice thickness and a regularization of the ice surface velocity can be defined as follows
```julia
loss = MultiLoss(
    losses=(LossH(), VelocityRegularization()),
    λs=(0.4, 1e-5)
)
```

This `loss` must then be provided to the `empirical_loss_function` argument of `UDEparameters`.

In the following sections, we introduce how to define empirical and regularization loss terms, respectively.

### Empirical loss functions

The empirical error can be as simple as the sum of squares of the error between model and observations, but it can also involve more complex cost functions.
The complete description of the different losses are available in their corresponding docstrings (see the [API](./api.md)) but we provide here a brief summary for each of them.

There are very simple types which are agnostic to the nature of the variables whose error is being computed (that is $H$ or $V$). These are:
- `L2Sum`: $L^2$ sum of the error inside the glacier.
- `LogSum`: Logarithmic sum of the ratio between ice surface velocities (see [morlighem_inversion_2013](@cite)).

These types which define very simple operations are used in more complex loss functions:
- `LossH`: Loss function over the ice thickness only.
- `LossV`: Loss function over the ice surface velocity only.
- `LossHV`: Loss function over both the ice thickness and ice surface velocity, which is an alternative to `MultiLoss`.

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
\sum_{j \leq J} \left( \hat H(t_j,x_j)-H(t_j,x_j) \right)^2
```
with $\hat H(t_j,x_j)$ the predicted ice thickness at time $t_j$ and on the node of the simulation grid $x_j$.

### Regularization

Regularizations are very common in inverse modelling as they help to constraint the possible solutions of the problem to physical reasonable values.
From a mathematical and computational perspective, regularization losses are just another type of loss that do not include contribution from observations (and then have no _empirical_ contribution to their value).

ODINN currently supports the following type of regularization, although the development of new regularization should be straightforward from the source code API:
- `TikhonovRegularization`: Very common in geophysical inversions. Given a linear operator $A$, this is given by the value of $\| A(S) \|_2^2$, where $S$ is some state variable (e.g., the ice thickness or ice surface velocity). Default choice in ODINN is the Laplacian operator $\nabla^2$.
- `InitialThicknessRegularization`: Penalizes large second order derivatives in the initial condition of the glacier when this is treated as an optimization variable. This builds on top of `TikhonovRegularization`.
- `VelocityRegularization`: Penalizes large second order derivatives in the simulated surface velocity of the glacier. This builds on top of `TikhonovRegularization`.
Regularization and empirical losses can be combined together to construct new form of regularizations.

## Logging

`ODINN.jl` provides useful statistics, such as the training loss history or the parameters history in the inversion objects.
The training statistics can also be inspected with [TensorBoard](https://www.tensorflow.org/tensorboard) through VSCode or a local webserver.
You can either use the [TensorBoard VSCode extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.tensorboard) or simply install TensorBoard in a Python environment and then launch `tensorboard --logdir .logs/`.

By default the TensorBoard logs are stored in your ODINN folder in `ODINN/.logs/` but you may have to adapt the command above if you log in a different folder.
