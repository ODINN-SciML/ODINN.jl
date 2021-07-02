######################################################
###   Data structures for glacier hybrid models   ####
######################################################

using Base: @kwdef

# Machine learning training hyperameters
@kwdef mutable struct Hyperparameters
    batchsize::Int = 500     # batch size
    η::Float64 = 0.1         # learning rate
    epochs::Int = 20        # number of epochs
    use_cuda::Bool = true    # use gpu (if cuda available)
end


mutable struct Glacier
    bed::Array{Float64}    # bedrock height
    thick::Array{Float64}  # ice thickness
    vel::Array{Float64}    # surface velocities
    MB::Array{Float64}     # surface mass balance
    lat::Float64
    lon::Float64
end




