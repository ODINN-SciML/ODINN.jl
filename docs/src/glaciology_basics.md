# Glaciology basics

In this section we present a short summary of the main concepts used in this documentation, for readers without a glaciology background.

## What does ODINN model?

A glacier is described by its **ice thickness** ``H(x, y, t)`` on a grid. Together with the **bedrock** elevation ``B``, it gives the **surface elevation** ``S = B + H``. The evolution of the ice thickness follows from mass conservation:

```math
\frac{\partial H}{\partial t} = \dot{b} - \nabla \cdot \mathbf{q}
```

where ``\dot{b}`` is the **surface mass balance** and ``\mathbf{q}`` is the **ice flux**, the ice moving downhill under its own weight. Each term is handled by a different package: [`Muninn`](Packages/muninn.md) computes ``\dot{b}`` and [`Huginn`](Packages/huginn.md) computes ``\mathbf{q}``.

### Surface mass balance

The surface mass balance ``\dot{b}`` is the snow accumulation minus the melt at the glacier surface, in meters of water equivalent per year (m w.e. yr⁻¹). It is positive where the glacier gains mass (usually at high elevations) and negative where it loses mass. `ODINN.jl` includes temperature-index models, where melt is proportional to the positive air temperatures through a **degree-day factor** (DDF), which can be [calibrated](smb_calibration.md) against observations. It also supports machine learning models from [`MassBalanceMachine.jl`](Packages/massbalancemachine.md).

### Ice flow

The **ice flow model** computes the ice flux ``\mathbf{q}`` from ``H`` and ``S``. Ice moves because of two processes: the internal deformation of the ice and its sliding over the bedrock. We describe the deformation with Glen's flow law [cuffey_physics_2010](@cite):

```math
\dot{\varepsilon} = A \tau^{n}
```

where ``\dot{\varepsilon}`` is the strain rate, ``\tau`` is the stress, ``n`` is the flow law exponent (usually taken as 3 [cuffey_physics_2010](@cite)) and ``A`` is the creep coefficient, which controls how soft the ice is. The sliding is described by a sliding law, which relates the basal velocity ``u_b`` to the basal shear stress ``\tau_b`` and to the effective pressure ``N`` at the glacier bed (the ice pressure minus the water pressure):

```math
u_b = C \tau_b^{p} N^{-q}
```

where ``C`` is the sliding coefficient, and ``p`` and ``q`` are exponents (``q = 0`` corresponds to a Weertman-type sliding law).

Ice flow models differ in the physical approximations they make to compute ``\mathbf{q}`` from these two processes. `ODINN.jl` currently implements the Shallow Ice Approximation (SIA), but the equation above does not depend on this choice, and other models (e.g. SSA or DIVA) can be used in the same way, see [Extending ODINN](extending.md).

## Forward and inverse modelling

In a **forward simulation**, the model parameters and the initial state of the glacier are known, and we simulate how the glacier evolves, see the [Forward simulation](forward_simulation.md) tutorial. In an **inversion**, we go the other way around: we look for the parameters or the initial state that make the model match the observations.

There are two kinds of inversions, described in [Inversion types](inversions.md):

  - **Classical inversions** directly optimize a parameter of the model, such as the creep coefficient ``A`` or the initial ice thickness.
  - **Functional inversions** optimize a function instead. A **law** maps input variables (e.g. the air temperature) to a quantity used by the model (e.g. ``A``), and it can be learned with a regressor such as a neural network. The model then becomes a **Universal Differential Equation** (UDE), a differential equation where unknown terms are replaced by machine learning models.

Both minimize a loss function that measures the mismatch with the observations, see [Optimization](optimization.md), and they rely on gradients of the model, which are available because the ecosystem is differentiable, see [Sensitivity analysis](sensitivity.md).

## Glaciers and data

Glaciers are identified by their **RGI ID**, from the [Randolph Glacier Inventory](https://www.glims.org/RGI/) (RGI) version 6.0 [rgi_consortium_randolph_2017](@cite). For example, `RGI60-11.03638` (Argentière) is glacier number 03638 of region 11 (Central Europe) in version 6.0 of the inventory.

The data needed to run a simulation is preprocessed with [`Gungnir`](Packages/gungnir.md), using the Open Global Glacier Model (OGGM) of Maussion et al. (2019) [maussion_open_2019](@cite), see [Glaciers](glaciers.md). The main datasets we use are:

  - **Ice thickness**: the consensus estimate of Farinotti et al. (2019) [farinotti_consensus_2019](@cite), the ice thickness of Millan et al. (2022) [millan_ice_2022](@cite) and the observations of the Glacier Thickness Database (GlaThiDa) [welty_worldwide_2020](@cite), often measured with ground-penetrating radar (GPR).
  - **Surface velocities**: Millan et al. (2022) [millan_ice_2022](@cite).
  - **Geodetic mass balance**: the glacier-wide mass change from satellite observations of Hugonnet et al. (2021) [hugonnet_accelerated_2021](@cite), used to calibrate mass balance models.
  - **Climate**: W5E5 [lange_wfde5_2019](@cite) by default, and ERA5-Land [munoz_sabater_era5_land_2021](@cite).

## Glossary

  - **DDF**: degree-day factor, the amount of melt per positive degree day (PDD).
  - **DIVA**: Depth-Integrated Viscosity Approximation, an ice flow model that accounts for longitudinal and lateral stresses, which matters for fast-flowing and sliding-dominated glaciers.
  - **dh/dt**: rate of change of the glacier surface elevation, observed from satellites.
  - **m w.e.**: meters of water equivalent, the unit of mass balance.
  - **MB** and **SMB**: mass balance and surface mass balance.
  - **SciML**: scientific machine learning, the combination of physical models and machine learning.
  - **SSA**: Shallow Shelf Approximation, an ice flow model for fast-flowing, sliding-dominated ice.
