# Abstract type as a parent type for Mass Balance models
abstract type MBmodel end

###############################################
########## TEMPERATURE-INDEX MODELS ###########
###############################################

# Subtype structure for Temperature-Index Mass Balance model
abstract type TImodel <: MBmodel end
# Temperature-index model with 1 melt factor
# Make these mutable if necessary
@kwdef struct TImodel1{F <: AbstractFloat} <: TImodel
    DDF::F
    acc_factor::F
end

"""
    TImodel1(;
        DDF::Float64 = 5.0,
        acc_factor::Float64 = 1.0
        )
Temperature-index model with a single degree-day factor.

Keyword arguments
=================
    - `DDF`: Single degree-day factor, for both snow and ice.
    - `acc_factor`: Accumulation factor
"""
function TImodel1(;
            DDF::Float64 = 5.0,
            acc_factor::Float64 = 1.0
            )

    # Build the simulation parameters based on input values
    TI_model = TImodel1(DDF, acc_factor)

    return TI_model
end

# Temperature-index model with 2 melt factors
@kwdef struct TImodel2{F <: AbstractFloat} <: TImodel
    DDF_snow::F
    DDF_ice::F
    acc_factor::F
end

"""
    TImodel2(;
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
function TImodel2(;
            DDF_snow::Float64 = 3.0,
            DDF_ice::Float64 = 6.0,
            acc_factor::Float64 = 1.0
            )

    # Build the simulation parameters based on input values
    TI_model = TImodel2(DDF_snow, DDF_ice, acc_factor)

    return TI_model
end

###############################################
################### UTILS #####################
###############################################

include("mass_balance_utils.jl")

