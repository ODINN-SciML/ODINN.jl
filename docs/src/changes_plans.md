# Ongoing changes and future plans

In this page we will attempt to document the main ongoing changes in terms of model development, and the main features we are planning for the future.

  - We are currently working on ensuring full end-to-end differentiability of the whole ODINN ecosystem with `SciMLSensitivy.jl` and `Enzyme.jl` to have automatic continuous adjoints. We are very close to achieving this. A new release, including other ongoing features will be announced once everything is properly integrated and tested.

  - We are interested in implementing other glacier equations inside ODINN (e.g., full Stokes equation, Shallow Shelf Equation). ODINN offers large flexibility and composability to easily integrate new differential equations that can utilize both the forward and inverse capabilities inside ODINN.
  - We have plans to host all the preprocessed glacier directories in a server, so users can automatically download them without having to preprocess them using `Gungnir`.
  - GPU compatibility is still not available. For now, we are focusing on having everything parallelized with multiprocessing while being compatible with automatic differentiation (AD). Once this codebase is stable, we might implement a GPU-compatible version.
