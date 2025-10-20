# Parameters

There are different types of parameters in ODINN, holding specific information for different modelling aspects. All the types of parameters are wrapped into a parent `Parameter` type, which is threaded throughout `ODINN.jl`.

```@docs
Sleipnir.Parameters
ODINN.Parameters()
```

The main child types of parameters are the following ones:

## Simulation parameters

Simulation parameters are used to specify anything related to ODINN simulations, ranging from types, working directories to multiprocessing.

```@docs
Sleipnir.SimulationParameters
Sleipnir.SimulationParameters()
```

## Physical parameters

Physical parameters are used to store physical constants and variables used in the physical and machine learning models.

```@docs
Sleipnir.PhysicalParameters
Sleipnir.PhysicalParameters()
```

## Solver parameters

Solver parameters determine all aspects related to the numerical scheme used to solve the differential equations of glacier ice flow.

```@docs
Huginn.SolverParameters
Huginn.SolverParameters()
```

## Hyperparameters

Hyperparameters determine different aspects of a given machine learning model. For now, these are focused on neural networks, but we plan to extend them in the future for other types of regressors.

```@docs
ODINN.Hyperparameters
ODINN.Hyperparameters()
```

## UDE parameters

Universal Differential Equation (UDE) parameters are used to determine different modelling choices regarding the use of UDEs, such as wich sensitivity algorithm or optimization method to use.

```@docs
ODINN.UDEparameters
ODINN.UDEparameters()
```