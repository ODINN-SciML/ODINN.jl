
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
    UA = Flux.f64(UA)
    # See if parameters need to be retrained or not
    θ, UA_f = Flux.destructure(UA)
    if !isnothing(θ_trained)
        θ = θ_trained
    end
    return UA, θ
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
function A_polyfit(A_values)
    return fit(A_values[1,:], A_values[2,:]) # degree = length(xs) - 1
end

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