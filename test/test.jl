# -*- coding: utf-8 -*-
# testing
import Pkg
Pkg.activate(dirname(Base.current_project())) # activate project
Pkg.precompile()
#Pkg.resolve()
# Pkg.instantiate()

using ODINN
using Test
# using Random
# using Distributed
# using Statistics, LinearAlgebra, Random, Polynomials
# using HDF5
# using JLD2
# using OrdinaryDiffEq, DiffEqFlux
# using Zygote: @ignore_derivatives
# using Flux
# using Tullio, RecursiveArrayTools
# using Infiltrator
# using Plots
# using ProgressMeter, ParallelDataTransfer
# using Dates
# using Makie, CairoMakie

# @everywhere begin 
#     import Pkg
#     Pkg.activate(dirname(Base.current_project()))
# end

# include all functions 
# include("../scripts/toy_model/helpers/parameters.jl")
# include("../test/utils_test.jl")
# include("../scripts/toy_model/helpers/iceflow.jl")

@testset "Halfar" begin

# ######################################################################################################
# ######################             Testing Forward SIA Model            ##############################
# ######################################################################################################


# # I CHANGED ICEFLOW TO REMOVE THE VELOCITIES!!! 

# I need to bypass Build_PDE_context. That is, create an object that looks like the output
# of this function and pass it to batch_iceflow_PDE



const nx, ny = 100, 100             # Definition of these constants inside the function? const doesn't work --> global
#t₀ = 1
#t₁ = 5
Δx, Δy = 50, 50
#nx, ny = 100, 100
A₀ = 1e-16
h₀ = 500        # [m]
r₀ = 1000       # [m]
ν = (A₀, h₀, r₀)

B = zeros((nx,ny))

# things from glacier_UDE
rng_seed() = MersenneTwister(123) # random seed
solver = Ralston()
#(@isdefined temp_series) || (const temp_series, norm_temp_series = fake_temp_series(t₁))

temps = [0,-2.0,-3.0,-5.0,-10.0,-12.0,-14.0,-15.0,-20.0]
const temp_series = [repeat([tmp], Int(t₁-t₀)+1) for tmp in temps]

# function SIA_solver_test()

    # this should run with different initializations 
    # cosntant were suppose to go here

H₀ = [ halfar_solution(t₀, sqrt((Δx * (i - nx/2))^2 + (Δy * (j - ny/2))^2), ν) for i in 1:nx, j in 1:ny]

H_refs = generate_ref_dataset(temp_series, H₀)   # we need to specify the time!!!

# ################ Non-negativity of solutions ##################

function non_negative_solutions(Hs)
    @testset "Non-negativity of solutions" begin
    for i in 1:length(Hs)
        @test 0.0 <= minimum(Hs[i])
    end
end
end

# ############### Conservation of mass ##################

function mass_conservation(H₀, Hs)
    @testset "Conservation of ice mass under zero mass balance conditions" begin
        for i in 1:length(Hs)
            @test norm(Hs[i],1) ≈ norm(H₀,1) atol=1e-2
        end
    end
end

# ############### Halfar Solutions ##################

function halfar_test(H₀, Hs)
    @testset "Halfar Solution Test" begin 
    for i in 1:length(H_refs)
        T_mean = mean(temp_series[i])
        println("Current temperature:", T_mean)
        A₁ = A_fake(T_mean)
        τ₁ = t₀ + (A₀ / A₁) * ( t₁ - t₀ )
        H₁ = [ halfar_solution(τ₁, sqrt((Δx * (i - nx/2))^2 + (Δy * (j - ny/2))^2), ν) for i in 1:nx, j in 1:ny];

        println("Norms")
        println( norm(H₀, 1))
        println( norm(H_refs[i],1))
        println( norm(H₁, 1))

        #display(plot(heatmap(H₀), heatmap(H₁), heatmap(H_refs[i]), heatmap(abs.(H₁ - H_refs[i])) ,layout=(2,2)))
        #display(heatmap!(H_refs[i]))

        abs_diff = maximum(abs.(H₁ .- H_refs[i]))

        println(abs_diff)
        @test abs_diff < 0.1
    end
    end
end




### All the test together
non_negative_solutions(H_refs)
mass_conservation(H₀, H_refs)
halfar_test(H₀, H_refs)



# # Test UA where this is just the identity function 

UA = FastChain(
    FastDense(1, 1, x->A_fake.(x))
    )
θ = initial_params(UA)

temp = temps[1]

UA(0.0, θ)

predict_A̅(UA, θ, temp_series[1])

end
