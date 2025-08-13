# Optimization

In this section, we discuss the aspects of the inversions related to parameter or model state optimization and the associated loss functions.

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

However, since we have very sparse observations, data is available only at specific points in time and space. The ground truth ice thickness, generally coming from ground penetrating radar (GPR) field work from the Glathida database [welty_worldwide_2020](@cite), displays a very strong sparsity, with observations concentrated along radar transects. Glacier ice surface velocity products , e.g. [millan_ice_2022](@cite), [rabatel_satellite-derived_2023](@cite), are notoriously less sparse, but still present many gaps in the grid due to signal-to-noise-ratio issues from the products. 
Let $\mathcal{X}=(x_j,t_j)_{j\leq J}$ define the set of points where there are ground truth measurements.
We assume that $\forall j\leq J,\, x_j\in(x_i)_i$, that is the ground truth measurements are aligned with the simulation grid.
The data fidelity term in the discrete setting then becomes
$$\mathcal{D}\left(\hat S, S \right) \stackrel{\mathrm{def}}{=} \sum_{j \leq J} \left( \hat H(t_j,x_j)-H(t_j,x_j) \right)^2$$
with $\hat H(t_j,x_j)$ the predicted ice thickness at time $t_j$ and on the node of the simulation grid $x_j$.

