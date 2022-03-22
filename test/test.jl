# testing
import Pkg
Pkg.activate(dirname(Base.current_project())) # activate project
#Pkg.resolve()
#Pkg.instantiate()

using Test
using Statistics
using LinearAlgebra
using Random
using HDF5  
using JLD2
using OrdinaryDiffEq
using DiffEqFlux
using Flux
using Tullio
using RecursiveArrayTools
using Infiltrator
using Plots
using ProgressMeter
using Distributed

# include all functions 
include("../scripts/toy_model/helpers/parameters.jl")
include("../scripts/toy_model/helpers/iceflow.jl")


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


# things from glacier_UDE
rng_seed() = MersenneTwister(123) # random seed
solver = Ralston()
(@isdefined temp_series) || (const temp_series, norm_temp_series = fake_temp_series(t₁))

## I CHANGED ICEFLOW TO REMOVE THE VELOCITIES!!! 

const nx, ny = 100, 100             # Definition of these constants inside the function? const doesn't work
t₀ = 1
t₁ = 5
Δx, Δy = 50, 50
#nx, ny = 100, 100
A₀ = 1e-16
h₀ = 500        # [m]
r₀ = 1000       # [m]
ν = (A₀, h₀, r₀)

B = zeros((nx,ny))

function SIA_solver_test()

    # this should run with different initializations 
    # cosntant were suppose to go here

    H₀ = [ halfar_solution(t₀, sqrt((Δx * (i - nx/2))^2 + (Δy * (j - ny/2))^2), ν) for i in 1:nx, j in 1:ny]

    H_refs = generate_ref_dataset(temp_series, H₀)   # we need to specify the time!!!

    @testset "Forward Model Test" begin 
    for i in 1:length(H_refs)
        T_mean = mean(temp_series[i])
        A_mean = A_fake(T_mean)
        τ₁ = (A_mean / A₀) * t₁
        H₁ = [ halfar_solution(τ₁, sqrt((Δx * (i - nx/2))^2 + (Δy * (j - ny/2))^2), ν) for i in 1:nx, j in 1:ny];
        abs_diff = maximum(abs.(H₁ .- H_refs[i]))
        println(abs_diff)
        @test abs_diff < 20
    end
    end

end 

SIA_solver_test()