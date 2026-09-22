# Ongoing changes and future plans

In this page we will attempt to document the main ongoing changes in terms of model development, and the main features we are planning for the future.

  - End-to-end differentiability of the whole ODINN ecosystem with `SciMLSensitivity.jl` and `Enzyme.jl` is now available through `SciMLSensitivityAdjoint`, alongside ODINN's own manual adjoints. We are continuing to broaden the set of targets and losses covered by the automatic path, and to consolidate it across the ecosystem.

  - We plan to add the Depth-Integrated Viscosity Approximation (DIVA) to `Huginn` and `ODINN` in the coming months, to improve on the physics of the Shallow Ice Approximation. Unlike the SIA, DIVA accounts for longitudinal and lateral stresses and resolves the velocity as part of the solution, which matters for fast-flowing and sliding-dominated glaciers. It will be usable with both the forward and the inverse capabilities of ODINN.

  - We are interested in implementing other glacier equations inside ODINN (e.g., full Stokes equation, Shallow Shelf Equation). ODINN offers large flexibility and composability to easily integrate new differential equations that can utilize both the forward and inverse capabilities inside ODINN.

  - GPU compatibility is still not available. For now, we are focusing on having everything parallelized with multiprocessing while being compatible with automatic differentiation (AD). Once this codebase is stable, we might implement a GPU-compatible version.
