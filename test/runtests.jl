import Pkg
Pkg.activate(dirname(Base.current_project()))
Pkg.precompile()

using ODINN
using Test
using Flux
using JLD, PyCallJLD

@testset "SIA PDE simulations" begin include("PDE_solve.jl") end
@testset "SIA UDE training" begin include("UDE_train.jl") end
