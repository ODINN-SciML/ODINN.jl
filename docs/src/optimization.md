# Optimization

In this section we present the main ideas behind the inversions.
In data assimilation, one aims at calibrating a model such that the predictions match the observations.
The observations can be either an initial state, a final state or observations over time.
The first two are called steady state inversions while the last one corresponds to a transient inversion.

In the standard settting, inversions are made with respect to some quantity of interest that is involed in the iceflow equation.
This can be for example the creep coefficient $A$ for the Shallow Ice Approximation (SIA) like in [bolibar_universal_2023](@cite).
This parameter can be constant for one glacier or vary with respect to space, in which case a spatial regularization is added to make the inversion problem well-posed.

Another way to invert a parameter in an iceflow equation is to parametrize the quantity of interest, let us say $A$, by other quantities which can be in the case of [bolibar_universal_2023](@cite), the long term air temperature $T$.
The optimized variable is not $A$ but the parameters $\theta$ and the mapping $(T,\theta)\mapsto A(\theta,T)$ defines a parametrization of the ice rheology.

We summarize the main differences between the two kind of inversions hereafter but first let us define the loss functions used in ODINN.

## Loss functions

The loss function that is being optimized $\mathcal{L}$ consists of a data fidelity term $\mathcal{D}$ and an optional regularization $\mathcal{R}$.
The data fidelity term represents the error between the predicted state of the glacier $\hat S$ and its ground truth state $S$. The state $S$ consists of the thickness $H$ and the ice surface velocity $V$ and we write $S=\begin{pmatrix}H\\ V\end{pmatrix}$. The ground truth for each of these variables is available through different datasets.

The data fidelity term to minimize is $$\mathcal{D}\left(\hat S, S \right)$$

and it can be as simple as the sum of two $L^2$ losses for each of the variables, or involve more complex cost functions.
The complete description of the different losses are available in their corresponding docstrings but we provide here a brief summary for each of them.

There are very simple types which are agnostic to the nature of the variables whose error is being computed (that is $H$ or $V$). These are:
- `L2Sum`: $L^2$ sum of the error inside the glacier.
- `LogSum`: Logarithmic sum used in [morlighem_inversion_2013](@cite).

These types which define very simple operations are used in more complex loss functions:
- `LossH`: Loss function over the ice thickness only.
- `LossV`: Loss function over the ice surface velocity only.
- `LossHV`: Loss function over both the ice thickness and ice surface velocity.

In the case of the $L^2$ loss, the error for the ice thickness is mathematically defined as:
$$\int_{t\in\mathcal{T}} \int_{x\in\Omega} \left(\hat H(t, x)-H(t, x)\right)_2^2\,\mathrm{d}t\,\mathrm{d}x$$
where $\Omega\subset\mathbb{R}^2$ defines the inner mask of the glacier where each pixel is at least at a given distance from the borders.
In the formula above, $\hat H$ and $H$ are written in a continuous setting where no grid is defined.
In practice, the iceflow equation is solved on a given grid $(x_i)_{i\leq I}$ where each $x_i\in\mathbb{R}^2$.
However, because we have very sparse data, the ground truth ice thickness is available only at specific points in time and space.
Let $\mathcal{X}=(x_j,t_j)_{j\leq J}$ define the set of points where there are ground truth measurements.
We assume that $\forall j\leq J,\, x_j\in(x_i)_i $, that is the ground truth measurements are aligned with the simulation grid.
The data fidelity term in the discrete setting then becomes
$$\mathcal{D}\left(\hat S, S \right) \stackrel{\mathrm{def}}{=} \sum_{j\leq J} \left( \hat H(t_j,x_j)-H(t_j,x_j) \right)^2$$
with $\hat H(t_j,x_j)$ the predicted ice thickness at time $t_j$ and on the node of the simulation grid $x_j$.

## Classical inversions

The optimization problem is
$$\min_p \mathcal{D}\left(\hat S, S \right) + \mathcal{R}\left( \hat S, p \right)$$
where $p$ is the vector of parameters to invert, for example $p=[A]$.

## Functional inversions

We present the concept of functional inversion in the case where we want to learn a law for the ice rheology $A$ by using a neural network as a parametrization $A=\text{NN}(\theta,T)$ with weights $\theta$.
$$\begin{aligned}& \min_\theta \mathcal{D}\left(\hat S, S \right) + \mathcal{R}\left( \hat S, p \right)\\
& A=\text{NN}(\theta,T)
\end{aligned}$$
The functional inversion tutorial gives an exemple of how such inversion can be made in practice with `ODINN.jl`.
