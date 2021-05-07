######################################################
###   Data structures for glacier hybrid models   ####
######################################################

using Base: @kwdef

# Machine learning training hyperameters
@kwdef mutable struct Hyperparameters
    batchsize::Int = 500     # batch size
    η::Float64 = 0.1         # learning rate
    epochs::Int = 500        # number of epochs
    use_cuda::Bool = true    # use gpu (if cuda available)
end


mutable struct Glacier
    bed::Array{Float32}    # bedrock height
    thick::Array{Float32}  # ice thickness
    vel::Array{Float32}    # surface velocities
    MB::Array{Float32}     # surface mass balance
    lat::Float32
    lon::Float32
end

# Grid initialization
dSdx    = zeros(nx-1, ny  )
dSdy    = zeros(nx  , ny-1)
∇S      = zeros(nx-1, ny-1)
D       = zeros(nx-1, ny-1)
Fx      = zeros(nx-1, ny-2)
Fy      = zeros(nx-2, ny-1)
F       = zeros(nx-2, ny-2)
dHdt    = zeros(nx-2, ny-2)
MB      = zeros(nx, ny);


