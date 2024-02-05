export run₀

"""
    run_inversion!(simulation::Inversion)

Out-place run of classic inversion model. 
"""
function run₀(simulation::Inversion)

    enable_multiprocessing(simulation.parameters)

    println("Running forward out-place PDE ice flow inversion model...\n")
    inversion_params_list = @showprogress pmap((glacier_idx) -> invert_iceflow(glacier_idx, simulation), 1:length(simulation.glaciers))

    simulation.inversion = inversion_params_list

    @everywhere GC.gc() # run garbage collector

end

function invert_iceflow(glacier_idx::Int, simulation::Inversion) 
    model = simulation.model
    params = simulation.parameters
    glacier = simulation.glaciers[glacier_idx]
    
    glacier_id = isnothing(glacier.gdir) ? "unnamed" : glacier.rgi_id
    println("Processing glacier: ", glacier_id)
    
    nx, ny = glacier.nx, glacier.ny 
    
    model.iceflow.A=glacier.A
    model.iceflow.n=glacier.n
    model.iceflow.glacier_idx = glacier_idx
    model.iceflow.H₀ = deepcopy(glacier.H₀)
    model.iceflow.H  = deepcopy(glacier.H₀)
    model.iceflow.MB = zeros(Float64,nx,ny)
    model.iceflow.MB_mask = zeros(Float64,nx,ny)
    model.iceflow.MB_total = zeros(Float64,nx,ny)

    params.solver.tstops = Huginn.define_callback_steps(params.simulation.tspan, params.solver.step)
    stop_condition(u,t,integrator) = Sleipnir.stop_condition_tstops(u,t,integrator, params.solver.tstops) #closure
    
    function action!(integrator)
        if params.simulation.use_MB 
            # Compute mass balance
            MB_timestep!(model, glacier, params.solver.step, integrator.t)
            apply_MB_mask!(integrator.u, glacier, model.iceflow)
        end
    end

    cb_MB = DiscreteCallback(stop_condition, action!)

    function SIA2D(H, p, t)
        simulation=p[1]
        x=p[2]
    
        # function SIA2D(H, SIA2Dmodel)
            # Retrieve parameters
            SIA2D_model = simulation.model.iceflow
            glacier = simulation.glaciers[simulation.model.iceflow.glacier_idx[]]
            params = simulation.parameters

            # Retrieve parameters
            B = glacier.B
            Δx = glacier.Δx
            Δy = glacier.Δy
            A = x[1]
            n = SIA2D_model.n
            ρ = params.physical.ρ
            g = params.physical.g

            # Update glacier surface altimetry
            S = B .+ H
            
            # All grid variables computed in a staggered grid
            # Compute surface gradients on edges
            dSdx = Huginn.diff_x(S) ./ Δx
            dSdy = Huginn.diff_y(S) ./ Δy
            ∇S = (Huginn.avg_y(dSdx).^2 .+ Huginn.avg_x(dSdy).^2).^((n[] - 1)/2) 
            
            Γ = 2.0 * A[] * 1e-17  * (ρ * g)^n[] / (n[]+2) # 1 / m^3 s 
            D = Γ .* Huginn.avg(H).^(n[] + 2) .* ∇S
        
            # Compute flux components
            @views dSdx_edges = Huginn.diff_x(S[:,2:end - 1]) ./ Δx
            @views dSdy_edges = Huginn.diff_y(S[2:end - 1,:]) ./ Δy
        
            # Cap surface elevaton differences with the upstream ice thickness to 
            # imporse boundary condition of the SIA equation
            # We need to do this with Tullio or something else that allow us to set indices.
            η₀ = 1.0
            dSdx_edges .= @views @. min(dSdx_edges,  η₀ * H[2:end, 2:end-1]/Δx)
            dSdx_edges .= @views @. max(dSdx_edges, -η₀ * H[1:end-1, 2:end-1]/Δx)
            dSdy_edges .= @views @. min(dSdy_edges,  η₀ * H[2:end-1, 2:end]/Δy)
            dSdy_edges .= @views @. max(dSdy_edges, -η₀ * H[2:end-1, 1:end-1]/Δy)
        
            Fx = .-Huginn.avg_y(D) .* dSdx_edges
            Fy = .-Huginn.avg_x(D) .* dSdy_edges 
        
            #  Flux divergence
            @tullio dH[i,j] := -(Huginn.diff_x(Fx)[pad(i-1,1,1),pad(j-1,1,1)] / Δx + Huginn.diff_y(Fy)[pad(i-1,1,1),pad(j-1,1,1)] / Δy) 
        
            return dH
    end
    
    realsol = glacier.H_glathida
    plot_glacier(glacier, "heatmaps", [:H_glathida])
    

    function objfun(x, simulation, realsol)
        
        iceflow_prob = ODEProblem{false}(SIA2D, model.iceflow.H, params.simulation.tspan, tstops=params.solver.tstops, (simulation,x))
        iceflow_sol = solve(iceflow_prob, 
                        params.solver.solver, 
                        callback=cb_MB, 
                        tstops=params.solver.tstops, 
                        reltol=params.solver.reltol, 
                        abstol=params.solver.reltol,
                        save_everystep=params.solver.save_everystep, 
                        progress=params.solver.progress, 
                        progress_steps=params.solver.progress_steps)

        
        sol = iceflow_sol.u[end]        
        map!(x -> ifelse(x>0.0,x,0.0), sol, sol)
        
        ofv = 0.0
        if any((s.retcode != :Success for s in iceflow_sol))
            ofv = 1e12
        else
            ofv = mean((sol[realsol .!= 0] .- realsol[realsol .!= 0]) .^ 2)
            
        end

        
        return ofv
    end

    fn(x, p) = objfun(x, p[1], p[2])

    
    lower_bound = [0.0085]    
    upper_bound = [8.0] 
    
    optfun = OptimizationFunction(fn, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfun, [4.9], (simulation, realsol), lb = lower_bound,ub = upper_bound)
    sol = solve(optprob, BFGS(), x_tol=1.0e-3, f_tol = 1e-3)

    optimized_n = 3.0
    optimized_C = 0.0

    inversion_params = InversionParams(sol[1], optimized_n, optimized_C)
    
    return inversion_params
end

