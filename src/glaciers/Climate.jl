

@kwdef mutable struct Climate2Dstep{F <: AbstractFloat} 
    temp::Matrix{F}
    PDD::Matrix{F}
    snow::Matrix{F}
    rain::Matrix{F}
    gradient::Ref{F}
    avg_gradient::Ref{F}
    x::Vector{F}
    y::Vector{F}
    ref_hgt::Ref{F}
end



@kwdef mutable struct Climate{F <: AbstractFloat} 
    raw_climate::PyObject # Raw climate dataset for the whole simulation
    # Buffers to avoid memory allocations
    climate_raw_step::Ref{PyObject} # Raw climate trimmed for the current step
    climate_step::Ref{PyObject} # Climate data for the current step
    climate_2D_step::Climate2Dstep # 2D climate data for the current step to feed to the MB model
    longterm_temps::Vector{F} # Longterm temperatures for the ice rheology
    avg_temps::Ref{PyObject} # Intermediate buffer for computing average temperatures
    avg_gradients::Ref{PyObject} # Intermediate buffer for computing average gradients
end


