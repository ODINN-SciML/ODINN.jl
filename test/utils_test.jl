# -*- coding: utf-8 -*-
"""
    halfar_solution(t, r, θ)

Returns the evaluation of the Halfar solutions for the SIA equation. 

Arguments:
    - t: time
    - r: radial distance. The solutions have polar symmetry around the center of origin
    - ν = (A, H₀, R₀) 
"""
function halfar_solution(t, r, ν)

    # parameters of Halfar solutions
    A, h₀, r₀ = ν 

    Γ = 2 * A * (ρ * g)^n / (n+2)
    τ₀ = (7/4)^3 * r₀^4 / ( 18 * Γ * h₀^7 )   # characteristic time

    if r₀ * (t/τ₀)^(1/18) <= r
        return 0.0
    else
        return h₀ * (τ₀/t)^(1/9) * ( 1 - ( (τ₀/t)^(1/18) * (r/r₀) )^(4/3) )^(3/7)
    end
end
