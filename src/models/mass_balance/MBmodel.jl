include("mass_balance_utils.jl")

# Abstract type as a parent type for Mass Balance models
abstract type MB_model end
# Subtype structure for Temperature-Index Mass Balance model
abstract type TI_model <: MB_model end
# Temperature-index model with 1 melt factor
# Make these mutable if necessary
@kwdef struct TI_model_1 <: TI_model
    DDF::Float64
    acc_factor::Float64
end

"""
    TI_model_1(;
        DDF::Float64 = 5.0,
        acc_factor::Float64 = 1.0
        )
Temperature-index model with a single degree-day factor.

Keyword arguments
=================
    - `DDF`: Single degree-day factor, for both snow and ice.
    - `acc_factor`: Accumulation factor
"""
function TI_model_1(;
            DDF::Float64 = 5.0,
            acc_factor::Float64 = 1.0
            )

    # Build the simulation parameters based on input values
    TI_model = TI_model_1(DDF, acc_factor)

    return TI_model
end

# Temperature-index model with 2 melt factors
@kwdef struct TI_model_2 <: TI_model
    DDF_snow::Float64
    DDF_ice::Float64
    acc_factor::Float64
end

"""
    TI_model_2(;
        DDF_snow::Float64 = 3.0,
        DDF_ice::Float64 = 6.0,
        acc_factor::Float64 = 1.0
        )
Temperature-index model with two melt factors, for snow and ice.

Keyword arguments
=================
    - `DDF_snow`: Degree-day factor for snow.
    - `DDF_ice`: Degree-day factor for ice.
    - `acc_factor`: Accumulation factor
"""
function TI_model_2(;
            DDF_snow::Float64 = 3.0,
            DDF_ice::Float64 = 6.0,
            acc_factor::Float64 = 1.0
            )

    # Build the simulation parameters based on input values
    TI_model = TI_model_2(DDF_snow, DDF_ice, acc_factor)

    return TI_model
end

