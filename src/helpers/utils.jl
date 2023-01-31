# Helper functions for the staggered grid

####  Non-allocating functions  ####

diff_x!(O, I, Δx) = @views @. O = (I[begin+1:end,:] - I[1:end-1,:]) / Δx

diff_y!(O, I, Δy) = @views @. O = (I[:,begin+1:end] - I[:,1:end - 1]) / Δy

avg!(O, I) = @views @. O = (I[1:end-1,1:end-1] + I[2:end,1:end-1] + I[1:end-1,2:end] + I[2:end,2:end]) * 0.25

avg_x!(O, I) = @views @. O = (I[1:end-1,:] + I[2:end,:]) * 0.5

avg_y!(O, I) = @views @. O = (I[:,1:end-1] + I[:,2:end]) * 0.5

####  Allocating functions  ####

"""
    avg(A)

4-point average of a matrix
"""
@views avg(A) = 0.25 .* ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )


"""
    avg_x(A)

2-point average of a matrix's X axis
"""
@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )

"""
    avg_y(A)

2-point average of a matrix's Y axis
"""
@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )

"""
    diff_x(A)

2-point differential of a matrix's X axis
"""
@views diff_x(A) = (A[begin + 1:end, :] .- A[1:end - 1, :])

"""
    diff_y(A)

2-point differential of a matrix's Y axis
"""
@views diff_y(A) = (A[:, begin + 1:end] .- A[:, 1:end - 1])

"""
    inn(A)

Access the inner part of the matrix (-2,-2)
"""
@views inn(A) = A[2:end-1,2:end-1]

"""
    inn1(A)

Access the inner part of the matrix (-1,-1)
"""
@views inn1(A) = A[1:end-1,1:end-1]

"""
fillNaN!(x, fill)

Convert empty matrix grid cells into fill value
"""
function fillNaN!(A, fill=zero(eltype(A)))
    for i in eachindex(A)
        @inbounds A[i] = ifelse(isnan(A[i]), fill, A[i])
    end
end

function fillNaN(A, fill=zero(eltype(A)))
    return @. ifelse(isnan(A), fill, A)
end

function fillZeros!(A, fill=NaN)
    for i in eachindex(A)
        @inbounds A[i] = ifelse(iszero(A[i]), fill, A[i])
    end
end

function fillZeros(A, fill=NaN)
    return @. ifelse(iszero(A), fill, A)
end

"""
    smooth!(A)

Smooth data contained in a matrix with one time step (CFL) of diffusion.
"""
@views function smooth!(A)
    A[2:end-1,2:end-1] .= A[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(A[:,2:end-1], dims=1), dims=1) .+ diff(diff(A[2:end-1,:], dims=2), dims=2))
    A[1,:]=A[2,:]; A[end,:]=A[end-1,:]; A[:,1]=A[:,2]; A[:,end]=A[:,end-1]
end

function smooth(A)
    A_smooth = A[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(A[:,2:end-1], dims=1), dims=1) .+ diff(diff(A[2:end-1,:], dims=2), dims=2))
    @tullio A_smooth_pad[i,j] := A_smooth[pad(i-1,1,1),pad(j-1,1,1)] # Fill borders 
    return A_smooth_pad
end

function reset_epochs()
    @everywhere @eval ODINN global current_epoch = 1
    @everywhere @eval ODINN global loss_history = []
end

function set_current_epoch(epoch)
    @everywhere @eval ODINN global current_epoch = $epoch
end

function make_plots(plots_i)
    @everywhere @eval ODINN global plots = $plots_i
end

function set_use_MB(use_MB_i)
    @everywhere @eval ODINN global use_MB = $use_MB_i
end

function set_run_spinup(run_spinup_i)
    @everywhere @eval ODINN global run_spinup = $run_spinup_i
end

function set_use_spinup(use_spinup_i)
    @everywhere @eval ODINN global use_spinup = $use_spinup_i
end

function set_create_ref_dataset(create_ref_dataset_i)
    @everywhere @eval ODINN global create_ref_dataset = $create_ref_dataset_i
end

function set_train(train_i)
    @everywhere @eval ODINN global train = $train_i
end

function set_retrain(retrain_i)
    @everywhere @eval ODINN global retrain = $retrain_i
end

function set_ice_thickness_source(it_source_i)
    @everywhere @eval ODINN global ice_thickness_source = $it_source_i
end

function get_gdir_refs(refs, gdirs)
    gdir_refs = []
    for (ref, gdir) in zip(refs, gdirs)
        push!(gdir_refs, Dict("RGI_ID"=>gdir.rgi_id,
                                "H"=>ref["H"],
                                "Vx"=>ref["Vx"],
                                "Vy"=>ref["Vy"],
                                "S"=>ref["S"],
                                "B"=>ref["B"]))
    end
    return gdir_refs
end

"""
    generate_batches(batch_size, UD, target, gdirs_climate_batches, gdir_refs, context_batches; gtd_grids=nothing, shuffle=true)

Generates batches for the UE inversion problem based on input data and feed them to the loss function.
"""
function generate_batches(batch_size, UD, target::String, gdirs_climate_batches, gdir_refs, context_batches; gtd_grids=nothing, shuffle=true)
    targets = repeat([target], length(gdirs_climate_batches))
    UDs = repeat([UD], length(gdirs_climate_batches))
    if isnothing(gtd_grids) 
        gtd_grids = repeat([nothing], length(gdirs_climate_batches))
        batches = (UDs, gdirs_climate_batches, gdir_refs, context_batches, gtd_grids, targets)
    else
        batches = (UDs, gdirs_climate_batches, gdir_refs, context_batches, gtd_grids, targets)
    end
    train_loader = Flux.Data.DataLoader(batches, batchsize=batch_size, shuffle=shuffle)

    return train_loader
end

"""
    generate_batches(batch_size, UA, gdirs_climate_batches, context_batches, gdir_refs, UDE_settings; shuffle=true))

Generates batches for the UDE problem based on input data and feed them to the loss function.
"""
function generate_batches(batch_size, UA, gdirs_climate_batches, context_batches, gdir_refs, UDE_settings; shuffle=true)
    UAs = repeat([UA], length(gdirs_climate_batches))
    UDE_settings_batches = repeat([UDE_settings], length(gdirs_climate_batches))
    batches = (UAs, gdirs_climate_batches, context_batches, gdir_refs, UDE_settings_batches)
    train_loader = Flux.Data.DataLoader(batches, batchsize=batch_size, shuffle=shuffle)

    return train_loader
end


"""
    get_NN()

Generates a neural network.
"""
function get_NN(θ_trained)
    UA = Chain(
        Dense(1,3, x->softplus.(x)),
        Dense(3,10, x->softplus.(x)),
        Dense(10,3, x->softplus.(x)),
        Dense(3,1, sigmoid_A)
    )
    # See if parameters need to be retrained or not
    θ, UA_f = Flux.destructure(UA)
    if !isempty(θ_trained)
        θ = θ_trained
    end
    return UA_f, θ
end

function get_NN_inversion(θ_trained, target)
    if target == "D"
        U, θ = get_NN_inversion_D(θ_trained)
    elseif target == "A"
        U, θ = get_NN_inversion_A(θ_trained)
    end
    return U, θ
end

function get_NN_inversion_A(θ_trained)
    UA = Chain(
        Dense(1,3, x->softplus.(x)),
        Dense(3,10, x->softplus.(x)),
        Dense(10,3, x->softplus.(x)),
        Dense(3,1, softplus)
    )
    # See if parameters need to be retrained or not
    θ, UA_f = Flux.destructure(UA)
    if !isempty(θ_trained)
        θ = θ_trained
    end
    return UA_f, θ
end

function get_NN_inversion_D(θ_trained)
    UD = Chain(
        Dense(3,20, x->softplus.(x)),
        Dense(20,15, x->softplus.(x)),
        Dense(15,10, x->softplus.(x)),
        Dense(10,5, x->softplus.(x)),
        Dense(5,1, softplus) # force diffusivity to be positive
    )
    # See if parameters need to be retrained or not
    θ, UD_f = Flux.destructure(UD)
    if !isempty(θ_trained)
        θ = θ_trained
    end
    return UD_f, θ
end

"""
    predict_A̅(UA_f, θ, temp)

Predicts the value of A with a neural network based on the long-term air temperature.
"""
function predict_A̅(UA_f, θ, temp)
    UA = UA_f(θ)
    return UA(temp) .* 1e-17
end

function sigmoid_A(x) 
    minA_out = 8.0e-3 # /!\     # these depend on predict_A̅, so careful when changing them!
    maxA_out = 8.0
    return minA_out + (maxA_out - minA_out) / ( 1.0 + exp(-x) )
end

function sigmoid_A_inv(x) 
    minA_out = 8.0e-4 # /!\     # these depend on predict_A̅, so careful when changing them!
    maxA_out = 8.0e2
    return minA_out + (maxA_out - minA_out) / ( 1.0 + exp(-x) )
end

# Convert Pythonian date to Julian date
function jldate(pydate)
    return Date(pydate.dt.year.data[1], pydate.dt.month.data[1], pydate.dt.day.data[1])
end

function save_plot(plot, path, filename)
    Plots.savefig(plot,joinpath(path,"png","$filename-$(current_epoch[]).png"))
    Plots.savefig(plot,joinpath(path,"pdf","epoch$(current_epoch[]).pdf"))
end

function generate_plot_folders(path)
    if !isdir(joinpath(path,"png")) || !isdir(joinpath(path,"pdf"))
        mkpath(joinpath(path,"png"))
        mkpath(joinpath(path,"pdf"))
    end
end

# Polynomial fit for Cuffey and Paterson data 
A_f = fit(A_values[1,:], A_values[2,:]) # degree = length(xs) - 1

"""
    A_fake(temp, noise=false)

Fake law establishing a theoretical relationship between ice viscosity (A) and long-term air temperature.
"""
function A_fake(temp, A_noise=nothing, noise=false)
    # A = @. minA + (maxA - minA) * ((temp-minT)/(maxT-minT) )^2
    A = A_f.(temp) # polynomial fit
    if noise[]
        A = abs.(A .+ A_noise)
    end
    return A
end

function build_D_features(H::Matrix, temp, ∇S)
    ∇S_flat = ∇S[inn1(H) .!= 0.0] # flatten
    H_flat = H[H .!= 0.0] # flatten
    T_flat = repeat(temp,length(H_flat))
    X = Flux.normalise(hcat(H_flat,T_flat,∇S_flat))' # build feature matrix
    return X
end

function build_D_features(H::Float64, temp::Float64, ∇S::Float64)
    X = Flux.normalise(hcat([H],[temp],[∇S]))' # build feature matrix
    return X
end

function predict_diffusivity(UD_f, θ, X)
    UD = UD_f(θ)
    return UD(X)[1,:]
end

"""
    config_training_state(θ_trained)

Configure training state with current epoch and its loss history. 
"""
function config_training_state(θ_trained)
    if length(θ_trained) == 0
        reset_epochs()
    else
        # Remove loss history from unfinished trainings
        deleteat!(loss_history, current_epoch:length(loss_history))
    end
end

"""
    update_training_state(batch_size, n_gdirs)
    
Update training state to know if the training has completed an epoch. 
If so, reset minibatches, update history loss and bump epochs.
"""
function update_training_state(l, batch_size, n_gdirs)
    # Update minibatch count and loss for the current epoch
    global current_minibatches += batch_size
    global loss_epoch += l
    if current_minibatches >= n_gdirs
        # Track evolution of loss per epoch
        push!(loss_history, loss_epoch)
        println("Epoch #$(current_epoch[]) - Loss $(loss_type[]): ", loss_epoch)
        # Bump epoch and reset loss and minibatch count
        global current_epoch += 1
        global current_minibatches = 0
        global loss_epoch = 0.0
    end
end

function get_default_UDE_settings()
    if use_MB[]    
        UDE_settings = Dict("reltol"=>1e-7, 
                            "solver"=>RDPK3Sp35(), 
                            "sensealg"=>InterpolatingAdjoint(autojacvec=ReverseDiffVJP())) # Currently just ReverseDiffVJP supports callbacks.
    else
        UDE_settings = Dict("reltol"=>1e-7,
                            "solver"=>RDPK3Sp35(),
                            "sensealg"=>InterpolatingAdjoint(autojacvec=ZygoteVJP())) 
    end
end

function get_default_training_settings!(gdirs_climate, UDE_settings=nothing, train_settings=nothing, 
                                        θ_trained=[], random_MB=nothing)
    if isnothing(UDE_settings)
        UDE_settings = get_default_UDE_settings()
    end

    if isnothing(train_settings)
        train_settings = (Adam(), 1, length(gdirs_climate[1])) # solver, epochs, batch size
    end

    #### Setup default parameters ####
    if length(θ_trained) == 0
        reset_epochs()
        global loss_history = []
    end

    # Don't use MB if not specified
    if isnothing(random_MB)
        ODINN.set_use_MB(false) 
    end

    return UDE_settings, train_settings
end

function plot_test_error(pred::Dict{String, Any}, ref::Dict{String, Any}, variable, rgi_id, atol; path=joinpath(ODINN.root_dir, "test/plots"))
    @assert (variable == "H") || (variable == "Vx") || (variable == "Vy") "Wrong variable for plots. Needs to be either `H`, `Vx` or `Vy`."
    if !isapprox(pred[variable], ref[variable], atol=atol)
        # @warn "Error found in PDE solve! Check plots in /test/plots⁄"
        if variable == "H"
            colour=:ice
        elseif variable == "Vx" || variable == "Vy"
            colour=:speed
        end
        PDE_plot = Plots.heatmap(pred[variable] .- ref[variable], title="$(variable): PDE simulation - Reference simulation", c=colour)
        Plots.savefig(PDE_plot,joinpath(path,"$(variable)_PDE_$rgi_id.pdf"))
    end
end

function plot_test_error(pred::Tuple, ref::Dict{String, Any}, variable, rgi_id, atol; path=joinpath(ODINN.root_dir, "test/plots"))
    @assert (variable == "H") || (variable == "Vx") || (variable == "Vy") "Wrong variable for plots. Needs to be either `H`, `Vx` or `Vy`."
    if variable == "H"
        idx=1
        colour=:ice
    elseif variable == "Vx" 
        idx=2
        colour=:speed
    elseif variable == "Vy"
        idx=3
        colour=:speed
    end
    if !isapprox(pred[idx], ref[variable], atol=atol)
        # @warn "Error found in PDE solve! Check plots in /test/plots⁄"
        UDE_plot = Plots.heatmap(pred[idx] .- ref[variable], title="$(variable): UDE simulation - Reference simulation", c=colour)
        Plots.savefig(UDE_plot,joinpath(path,"$(variable)_UDE_$rgi_id.pdf"))
    end
end
