# Ongoing changes and future plans

In this page we will attempt to document the main ongoing changes in terms of model development, and the main features we are planning for the future.

  - Changes in transient inversions that modify existing results:

      + Each observation now counts once, with the same weight (`weighting = :uniform`). Before, an observation was weighted by the time since the previous one, and the first one was ignored. Loss values and the weights `λ` tuned before change. Since the loss is larger, the first step of a plain `LBFGS()` is larger too, and can put a bounded parameter on its bound, where LBFGS stops: use the LBFGS of `default_optimizer()`.
      + `θ.IC` is stored divided by the maximum initial thickness of each glacier (`H₀_scale`). Use `evaluate_H₀` to get metres. A `θ.IC` saved by an older version is in metres, and must be divided by `H₀_scale` before it is used with this version.
      + The default optimizer is Adam and then LBFGS (`default_optimizer()`), with 25 epochs each. With a vector of optimizers, `epochs` needs one number per optimizer.
      + The surface velocity includes the sliding term of the flux (plug flow) and is placed on the grid of `H`, so the velocity losses change when `C > 0`.
      + `InitialThicknessRegularization()` applies at the start of each glacier. A `t₀` outside the time window of a glacier is an error.
      + With an ice thickness survey at most one mass balance step after the start, an inversion of the initial condition starts on the survey (`align_start!`).

  - End-to-end differentiability of the whole ODINN ecosystem with `SciMLSensitivity.jl` and `Enzyme.jl` is now available through `SciMLSensitivityAdjoint`, alongside ODINN's own manual adjoints. We are continuing to broaden the set of targets and losses covered by the automatic path, and to consolidate it across the ecosystem.

  - We plan to add the Depth-Integrated Viscosity Approximation (DIVA) to `Huginn` and `ODINN` in the coming months, to improve on the physics of the Shallow Ice Approximation. Unlike the SIA, DIVA accounts for longitudinal and lateral stresses and resolves the velocity as part of the solution, which matters for fast-flowing and sliding-dominated glaciers. It will be usable with both the forward and the inverse capabilities of ODINN.

  - We are interested in implementing other glacier equations inside ODINN (e.g., full Stokes equation, Shallow Shelf Equation). ODINN offers large flexibility and composability to easily integrate new differential equations that can utilize both the forward and inverse capabilities inside ODINN.

  - GPU compatibility is still not available. For now, we are focusing on having everything parallelized with multiprocessing while being compatible with automatic differentiation (AD). Once this codebase is stable, we might implement a GPU-compatible version.
