######################################################
###   Data structures for glacier hybrid models   ####
######################################################

using Base: @kwdef

# Machine learning training hyperameters
@kwdef mutable struct Hyperparameters
    batchsize::Int = 9    # batch size
    η::Float32 = 0.01         # learning rate
    epochs::Int = 10        # number of epochs
    #epochs_internal::Int = 1        # number of epochs per single glacier
    use_cuda::Bool = true    # use gpu (if cuda available)
end


struct Glacier
    bed::Array{Float32}    # bedrock height
    thick::Array{Float32}  # ice thickness
    vel::Array{Float32}    # surface velocities
    MB::Array{Float32}     # surface mass balance
    lat::Float32
    lon::Float32
end




