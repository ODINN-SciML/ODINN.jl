
@everywhere begin
    @views avg(A) = 0.25 .* ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )
    @views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )
    @views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )
    @views diff_x(A) = (A[begin + 1:end, :] .- A[1:end - 1, :])
    @views diff_y(A) = (A[:, begin + 1:end] .- A[:, 1:end - 1])
    @views inn(A) = A[2:end-1,2:end-1]
    end # @everywhere 
    
    function glacier_evolution(glacier_list, ensemble=ensemble)
    
        # Initialize all matrices for the solver
        S, dSdx, dSdy = zeros(Float64,nx,ny),zeros(Float64,nx-1,ny),zeros(Float64,nx,ny-1)
        dSdx_edges, dSdy_edges, âS = zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1),zeros(Float64,nx-1,ny-1)
        H, D, dH, Fx, Fy = zeros(Float64,nx,ny), zeros(Float64,nx-1,ny-1),zeros(Float64,nx-2,ny-2),zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1)
        V, Vx, Vy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1)
        
        # Gather simulation parameters
        current_year = 0
        context = ArrayPartition([A], B, S, dSdx, dSdy, D, dSdx_edges, dSdy_edges, âS, Fx, Fy, Vx, Vy, V, [current_year])
    
        function prob_iceflow_func(prob, i, repeat, context, glacier_list) # closure
    
            H = glacier_gd[i].distributed_thickness.data # initial ice thickness conditions for forward model
            B = glacier_gd[i].topo.data - glacier_gd.distributed_thickness.data # bedrock
            
            println("Processing glacier #$i â ", mean(temp_series[i]))
            context.x[2] .= B # Bedrock for current glacier
    
            return remake(prob, p=context)
        end
    
        prob_func(prob, i, repeat) = prob_iceflow_func(prob, i, repeat, context, temp_series) # closure
    
        # Perform reference simulation with forward model 
        println("Running forward PDE ice flow model...\n")
        iceflow_prob = ODEProblem(iceflow!,H,(0.0,tâ),context)
        ensemble_prob = EnsembleProblem(iceflow_prob, prob_func = prob_func)
        iceflow_sol = solve(ensemble_prob, ROCK4(), ensemble, trajectories = length(temp_series), 
                            pmap_batch_size=length(temp_series), reltol=1e-6, 
                            progress=true, saveat=1.0, progress_steps = 50)
    
        # Save only matrices
        idx = 1
        for result in iceflow_sol
            if idx == 1
                H_refs = result.u[end]
            else
                @views H_refs = cat(H_refs, result.u[end], dims=3)
            end
            idx += 1
        end
    
        return H_refs  
    end
    
    function generate_ref_dataset(temp_series, Hâ, ensemble=ensemble)
        # Compute reference dataset in parallel
        H = deepcopy(Hâ)
        
        # Initialize all matrices for the solver
        S, dSdx, dSdy = zeros(Float64,nx,ny),zeros(Float64,nx-1,ny),zeros(Float64,nx,ny-1)
        dSdx_edges, dSdy_edges, âS = zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1),zeros(Float64,nx-1,ny-1)
        D, dH, Fx, Fy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-2,ny-2),zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1)
        V, Vx, Vy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1)
        A = 2e-16
        Îħ = 0                       # Weertman-type basal sliding (Weertman, 1964, 1972). 1 -> sliding / 0 -> no sliding
        C = 15e-14                  # Sliding factor, between (0 - 25) [mâ¸ NâğÂ³ aâğÂı]
        
        # Gather simulation parameters
        current_year = 0
        context = ArrayPartition([A], B, S, dSdx, dSdy, D, copy(temp_series[5]), dSdx_edges, dSdy_edges, âS, Fx, Fy, Vx, Vy, V, C, Îħ, [current_year])
    
        function prob_iceflow_func(prob, i, repeat, context, temp_series) # closure
            
            println("Processing temp series #$i â ", mean(temp_series[i]))
            context.x[7] .= temp_series[i] # We set the temp_series for the ith trajectory
    
            return remake(prob, p=context)
        end
    
        prob_func(prob, i, repeat) = prob_iceflow_func(prob, i, repeat, context, temp_series) # closure
    
        # Perform reference simulation with forward model 
        println("Running forward PDE ice flow model...\n")
        iceflow_prob = ODEProblem(iceflow!,H,(0.0,tâ),context)
        ensemble_prob = EnsembleProblem(iceflow_prob, prob_func = prob_func)
    
                    iceflow_sol = solve(ensemble_prob, ROCK4(), ensemble, trajectories = n_trajectories, 
                            pmap_batch_size=length(temp_series), reltol=1e-6, save_everystep=false, 
                            progress=true, saveat=1.0, progress_steps = 50)
    
        # Distributed progress bars for solving PDEs in paralell
        # @sync begin
        #     @async begin
        #         tasksdone = 0
        #         while tasksdone < n_trajectories
        #             tasksdone += take!(channel)
        #             update!(progress, tasksdone)
        #         end
        #     end
        #     @async begin
        #         iceflow_sol = solve(ensemble_prob, ROCK4(), ensemble, trajectories = n_trajectories, 
        #                     pmap_batch_size=length(temp_series), reltol=1e-6, save_everystep=false, 
        #                     progress=true, saveat=1.0, progress_steps = 50)
        #     end
        # end
    
        # Save only matrices
        idx = 1
        H_refs = [] 
        for result in iceflow_sol
            if idx == 1
                H_refs = result.u[end]
            else
                @views H_refs = cat(H_refs, result.u[end], dims=3)
            end
            idx += 1
        end
    
        return H_refs  
    end
    
    function train_iceflow_UDE(Hâ, UA, Î¸, train_settings, H_refs, temp_series)
        H = deepcopy(Hâ)
        current_year = 0.0
        optimizer = train_settings[1]
        epochs = train_settings[2]
        # Tuple with all the temp series and H_refs
        context = (B, H, current_year, temp_series)
        loss(Î¸) = loss_iceflow(Î¸, context, UA, H_refs) # closure
    
        # Debugging
        # println("Gradients: ", gradient(loss, Î¸))
        # @infiltrate
    
        println("Training iceflow UDE...")
        # println("Using solver: ", solver)
        iceflow_trained = DiffEqFlux.sciml_train(loss, Î¸, optimizer, cb=callback, maxiters = epochs)
    
        return iceflow_trained
    end
    
    @everywhere begin 
    
    callback = function (Î¸,l) # callback function to observe training
        println("Epoch #$current_epoch - Loss H: ", l)
    
        pred_A = predict_AÌ(UA, Î¸, collect(-20.0:0.0)')
        pred_A = [pred_A...] # flatten
        true_A = A_fake(-20.0:0.0)
    
        plot(true_A, label="True A")
        plot_epoch = plot!(pred_A, label="Predicted A")
        savefig(plot_epoch,joinpath(root_plots,"training","epoch$current_epoch.png"))
        global current_epoch += 1
    
        false
    end
    
    function loss_iceflow(Î¸, context, UA, H_refs) 
        H_preds = predict_iceflow(Î¸, UA, context)
        
        # Compute loss function for the full batch
        l_H = 0.0
    
        for i in 1:length(H_preds)
            H_ref = H_refs[:,:,i]
            H = H_preds[i].u[end]
            l_H += Flux.Losses.mse(H[H .!= 0.0], H_ref[H.!= 0.0]; agg=mean)
        end
    
        l_H_avg = l_H/length(H_preds)
        
        return l_H_avg
    end
    
    function output_func(sol, i)
        put!(channel, 1)
        sol, false
    end
    
    
    function predict_iceflow(Î¸, UA, context, ensemble=ensemble)
    
        function prob_iceflow_func(prob, i, repeat, context, UA) # closure
            # B, H, current_year, temp_series)  
            temp_series = context[4]
        
            # println("Processing temp series #$i â ", mean(temp_series[i]))
            # We add the ith temperature series 
            iceflow_UDE_batch(H, Î¸, t) = iceflow_NN(H, Î¸, t, context, temp_series[i], UA) # closure
            
            return remake(prob, f=iceflow_UDE_batch)
        end
    
        prob_func(prob, i, repeat) = prob_iceflow_func(prob, i, repeat, context, UA)
    
        # (B, H, current_year, temp_series)
        H = context[2]
        tspan = (0.0,tâ)
    
        iceflow_UDE(H, Î¸, t) = iceflow_NN(H, Î¸, t, context, temp_series[5], UA) # closure
        iceflow_prob = ODEProblem(iceflow_UDE,H,tspan,Î¸)
        ensemble_prob = EnsembleProblem(iceflow_prob, prob_func = prob_func)
        # ensemble_prob = EnsembleProblem(iceflow_prob, prob_func = prob_func, output_func = output_func)
    
        H_pred = solve(ensemble_prob, solver, ensemble, trajectories = length(temp_series), 
            pmap_batch_size=batch_size, u0=H, p=Î¸, reltol=1e-6, save_everystep=false,   
            progress=true, progress_steps = 10)
    
        # Distributed progress bars for solving UDEs in paralell
        # @sync begin
        #     @async begin
        #         tasksdone = 0
        #         println("tasksdone: ", tasksdone)
        #         while tasksdone < n_trajectories
        #             println("here")
        #             tasksdone += take!(channel)
        #             println("there")
        #             update!(progress, tasksdone)
        #             println("out")
        #         end
        #     end
        #     @async begin
        #         println("solving")
        #         H_pred = solve(ensemble_prob, solver, ensemble, trajectories = length(temp_series), 
        #                 pmap_batch_size=batch_size, u0=H, p=Î¸, reltol=1e-6, 
        #                 sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP()), save_everystep=false,   
        #                 progress=true, progress_steps = 10)
        #     end
        # end
    
        return H_pred
    end
    
    function iceflow!(dH, H, context,t)
        # Unpack parameters
        #A, B, S, dSdx, dSdy, D, temps, dSdx_edges, dSdy_edges, âS, Fx, Fy, Vx, Vy, V, C, Îħ, current_year 
        current_year = Ref(context.x[18])
        A = Ref(context.x[1])
        
        # Get current year for MB and ELA
        year = floor(Int, t) + 1
        if year != current_year[] && year <= tâ
            temp = Ref{Float64}(context.x[7][year])
            A[] .= A_fake(temp[])
            current_year[] .= year
        end
    
        # Compute the Shallow Ice Approximation in a staggered grid
        SIA!(dH, H, context)
    end    
    
    function iceflow_NN(H, Î¸, t, context, temps, UA)
    
        year = floor(Int, t) + 1
        if year <= tâ
            temp = temps[year]
        else
            temp = temps[year-1]
        end
    
        A = predict_AÌ(UA, Î¸, [temp]) # FastChain prediction requires explicit parameters
    
        # Compute the Shallow Ice Approximation in a staggered grid
        return SIA(H, A, context)
    end  
    
    """
        SIA!(dH, H, context)
    
    Compute a step of the Shallow Ice Approximation PDE in a forward model
    """
    function SIA!(dH, H, context)
        # Retrieve parameters
        #A, B, S, dSdx, dSdy, D, norm_temps, dSdx_edges, dSdy_edges, âS, Fx, Fy, Vx, Vy, V, C, Îħ, current_year, H_ref, H, UA, Î¸
        A = context.x[1]
        B = context.x[2]
        S = context.x[3]
        dSdx = context.x[4]
        dSdy = context.x[5]
        D = context.x[6]
        dSdx_edges = context.x[8]
        dSdy_edges = context.x[9]
        âS = context.x[10]
        Fx = context.x[11]
        Fy = context.x[12]
        
        # Update glacier surface altimetry
        S .= B .+ H
    
        # All grid variables computed in a staggered grid
        # Compute surface gradients on edges
        dSdx .= diff_x(S) / Îx
        dSdy .= diff_y(S) / Îy
        âS .= (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 
    
        Î = 2 * A * (Ï * g)^n / (n+2) # 1 / m^3 s 
        D .= Î .* avg(H).^(n + 2) .* âS
    
        # Compute flux components
        dSdx_edges .= diff_x(S[:,2:end - 1]) / Îx
        dSdy_edges .= diff_y(S[2:end - 1,:]) / Îy
        Fx .= .-avg_y(D) .* dSdx_edges
        Fy .= .-avg_x(D) .* dSdy_edges 
    
        #  Flux divergence
        inn(dH) .= .-(diff_x(Fx) / Îx .+ diff_y(Fy) / Îy) # MB to be added here 
    end
    
    # Function without mutation for Zygote, with context as an ArrayPartition
    function SIA(H, A, context)
        # Retrieve parameters
        B = context[1]
    
        # Update glacier surface altimetry
        S = B .+ H
    
        # All grid variables computed in a staggered grid
        # Compute surface gradients on edges
        dSdx = diff_x(S) / Îx
        dSdy = diff_y(S) / Îy
        âS = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 
    
        Î = 2 * A * (Ï * g)^n / (n+2) # 1 / m^3 s 
        D = Î .* avg(H).^(n + 2) .* âS
    
        # Compute flux components
        dSdx_edges = diff_x(S[:,2:end - 1]) / Îx
        dSdy_edges = diff_y(S[2:end - 1,:]) / Îy
        Fx = .-avg_y(D) .* dSdx_edges
        Fy = .-avg_x(D) .* dSdy_edges 
    
        #  Flux divergence
        @tullio dH[i,j] := -(diff_x(Fx)[pad(i-1,1,1),pad(j-1,1,1)] / Îx + diff_y(Fy)[pad(i-1,1,1),pad(j-1,1,1)] / Îy) # MB to be added here 
    
        return dH
    end
      
    function A_fake(temp, noise=false)
        A = @. minA + (maxA - minA) * ((temp-minT)/(maxT-minT) )^2
        if noise
            A = A .+ randn(rng_seed(), length(temp)).*4e-17
        end
    
        return A
    end
    
    predict_AÌ(UA, Î¸, temp) = UA(temp, Î¸) .* 1e-16
    
    function fake_temp_series(t, means=Array{Float64}([0,-2.0,-3.0,-5.0,-10.0,-12.0,-14.0,-15.0,-20.0]))
        temps, norm_temps, norm_temps_flat = [],[],[]
        for mean in means
           push!(temps, mean .+ rand(t).*1e-1) # static
           append!(norm_temps_flat, mean .+ rand(t).*1e-1) # static
        end
    
        # Normalise temperature series
        norm_temps_flat = Flux.normalise([norm_temps_flat...]) # requires splatting
    
        # Re-create array of arrays 
        for i in 1:tâ:length(norm_temps_flat)
            push!(norm_temps, norm_temps_flat[i:i+(tâ-1)])
        end
    
        return temps, norm_temps
    end
    
    end # @everywhere 
    