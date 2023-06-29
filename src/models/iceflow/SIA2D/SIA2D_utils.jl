

"""
SIA!(dH, H, SIA2Dmodel)

Compute an in-place step of the Shallow Ice Approximation PDE in a forward model
"""
function SIA2D!(dH::Matrix{F}, H::Matrix{F}, simulation::SIM, t::F) where {F <: AbstractFloat, SIM <: Simulation}
    
    # Retrieve parameters
    @timeit get_timer("ODINN") "Variable assignment" begin
    SIA2D_model::SIA2Dmodel = simulation.model.iceflow
    glacier::Glacier = simulation.glaciers[simulation.model.iceflow.glacier_idx[]]
    params::Parameters = simulation.parameters
        int_type = simulation.parameters.simulation.int_type
    H̄::Matrix{F} = SIA2D_model.H̄
    A::Ref{F} = SIA2D_model.A
    B::Matrix{F} = glacier.B
    S::Matrix{F} = SIA2D_model.S
    dSdx::Matrix{F} = SIA2D_model.dSdx
    dSdy::Matrix{F} = SIA2D_model.dSdy
    D::Matrix{F} = SIA2D_model.D
    Dx::Matrix{F} = SIA2D_model.Dx
    Dy::Matrix{F} = SIA2D_model.Dy
    dSdx_edges::Matrix{F} = SIA2D_model.dSdx_edges
    dSdy_edges::Matrix{F} = SIA2D_model.dSdy_edges
    ∇S::Matrix{F} = SIA2D_model.∇S
    ∇Sx::Matrix{F} = SIA2D_model.∇Sx
    ∇Sy::Matrix{F} = SIA2D_model.∇Sy
    Fx::Matrix{F} = SIA2D_model.Fx
    Fy::Matrix{F} = SIA2D_model.Fy
    Fxx::Matrix{F} = SIA2D_model.Fxx
    Fyy::Matrix{F} = SIA2D_model.Fyy
    Δx::F = glacier.Δx
    Δy::F = glacier.Δy
    Γ::Ref{F} = SIA2D_model.Γ
    n::int_type = simulation.parameters.physical.n
    ρ::F = simulation.parameters.physical.ρ
    g::F = simulation.parameters.physical.g
    end

    @timeit get_timer("ODINN") "H to zero" begin
    # First, enforce values to be positive
    map!(x -> ifelse(x>0.0,x,0.0), H, H)
    # Update glacier surface altimetry
    S .= B .+ H
    end

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    @timeit get_timer("ODINN") "Surface gradients" begin
    diff_x!(dSdx, S, Δx)  
    diff_y!(dSdy, S, Δy) 
    avg_y!(∇Sx, dSdx)
    avg_x!(∇Sy, dSdy)
    ∇S .= (∇Sx.^2 .+ ∇Sy.^2).^((n - 1)/2) 
    end

    @timeit get_timer("ODINN") "Diffusivity" begin
    # @infiltrate
    avg!(H̄, H)
    Γ[] = 2.0 * A[] * (ρ * g)^n / (n+2) # 1 / m^3 s 
    D .= Γ[] .* H̄.^(n + 2) .* ∇S
    end

    # Compute flux components
    @timeit get_timer("ODINN") "Gradients edges" begin
    @views diff_x!(dSdx_edges, S[:,2:end - 1], Δx)
    @views diff_y!(dSdy_edges, S[2:end - 1,:], Δy)
    end
    # Cap surface elevaton differences with the upstream ice thickness to 
    # imporse boundary condition of the SIA equation
    @timeit get_timer("ODINN") "Capping flux" begin
    η₀ = params.physical.η₀
    dSdx_edges .= @views @. min(dSdx_edges,  η₀ * H[1:end-1, 2:end-1]/Δx,  η₀ * H[2:end, 2:end-1]/Δx)
    dSdy_edges .= @views @. min(dSdy_edges,  η₀ * H[2:end-1, 1:end-1]/Δy,  η₀ * H[2:end-1, 2:end]/Δy) 
    dSdx_edges .= @views @. max(dSdx_edges, -η₀ * H[1:end-1, 2:end-1]/Δx, -η₀ * H[2:end, 2:end-1]/Δx)
    dSdy_edges .= @views @. max(dSdy_edges, -η₀ * H[2:end-1, 1:end-1]/Δy, -η₀ * H[2:end-1, 2:end]/Δy)
    end

    @timeit get_timer("ODINN") "Flux" begin
    avg_y!(Dx, D)
    avg_x!(Dy, D)
    Fx .= .-Dx .* dSdx_edges
    Fy .= .-Dy .* dSdy_edges 
    end

    #  Flux divergence
    @timeit get_timer("ODINN") "dH" begin
    diff_x!(Fxx, Fx, Δx)
    diff_y!(Fyy, Fy, Δy)
    inn(dH) .= .-(Fxx .+ Fyy) 
    end
end

# Dummy function to bypass ice flow
function noSIA2D!(dH::Matrix{F}, H::Matrix{F}, simulation::SIM, t::F) where {F <: AbstractFloat, SIM <: Simulation}
   
end

"""
    SIA(H, A, context)

Compute a step of the Shallow Ice Approximation UDE in a forward model. Allocates memory.
"""
function SIA2D(H, SIA2Dmodel)
    # Retrieve parameters
    B = SIA2Dmodel.B
    Δx = SIA2Dmodel.Δx
    Δy = SIA2Dmodel.Δy
    A = SIA2Dmodel.A

    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) ./ Δx
    dSdy= diff_y(S) ./ Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 

    Γ = 2.0 * A * (ρ * g)^n / (n+2) # 1 / m^3 s 
    D = Γ .* avg(H).^(n + 2) .* ∇S

    # Compute flux components
    @views dSdx_edges = diff_x(S[:,2:end - 1]) ./ Δx
    @views dSdy_edges = diff_y(S[2:end - 1,:]) ./ Δy

    # Cap surface elevaton differences with the upstream ice thickness to 
    # imporse boundary condition of the SIA equation
    # We need to do this with Tullio or something else that allow us to set indices.
    η₀ = 1.0
    dSdx_edges .= min.(dSdx_edges,  η₀ * H[1:end-1, 2:end-1]./Δx,  η₀ * H[2:end, 2:end-1]./Δx)
    dSdy_edges .= min.(dSdy_edges,  η₀ * H[2:end-1, 1:end-1]./Δy,  η₀ * H[2:end-1, 2:end]./Δy) 
    dSdx_edges .= max.(dSdx_edges, -η₀ * H[1:end-1, 2:end-1]./Δx, -η₀ * H[2:end, 2:end-1]./Δx)
    dSdy_edges .= max.(dSdy_edges, -η₀ * H[2:end-1, 1:end-1]./Δy, -η₀ * H[2:end-1, 2:end]./Δy)

    Fx = .-avg_y(D) .* dSdx_edges
    Fy = .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    @tullio dH[i,j] := -(diff_x(Fx)[pad(i-1,1,1),pad(j-1,1,1)] / Δx + diff_y(Fy)[pad(i-1,1,1),pad(j-1,1,1)] / Δy) 

    return dH
end

"""
    avg_surface_V(context, H, temp, sim, θ=[], UA_f=[])

Computes the average ice surface velocity for a given glacier evolution period
based on the initial and final ice thickness states. 
"""
function avg_surface_V!(H₀::Matrix{F}, simulation::SIM) where {F <: AbstractFloat, SIM <: Simulation}
    # We compute the initial and final surface velocity and average them
    # TODO: Add more datapoints to better interpolate this
    ft = simulation.parameters.simulation.float_type
    iceflow_model = simulation.model.iceflow
    Vx₀::Matrix{ft}, Vy₀::Matrix{ft} = surface_V!(H₀, simulation)
    Vx::Matrix{ft}, Vy::Matrix{ft} = surface_V!(iceflow_model.H, simulation)
    iceflow_model.Vx .= (Vx₀ .+ Vx)./2.0
    iceflow_model.Vy .= (Vy₀ .+ Vy)./2.0
    iceflow_model.V .= (iceflow_model.Vx.^2 .+ iceflow_model.Vy.^2).^(1/2) 
end

"""
    avg_surface_V(context, H, temp, sim, θ=[], UA_f=[])

Computes the average ice surface velocity for a given glacier evolution period
based on the initial and final ice thickness states. 
"""
function avg_surface_V(ifm::IF, temp::F, sim, θ=nothing, UA_f=nothing; testmode=false) where {F <: AbstractFloat, IF <: IceflowModel}
    # TODO: see how to get this
    # A_noise = parameters.A_noise
    # Δx, Δy, A_noise = retrieve_context(context, sim)

    # We compute the initial and final surface velocity and average them
    # TODO: Add more datapoints to better interpolate this
    Vx₀, Vy₀ = surface_V(ifm, sim, A_noise, θ, UA_f; testmode=testmode)
    Vx, Vy = surface_V(ifm, temp, sim, A_noise, θ, UA_f; testmode=testmode)
    V̄x = (Vx₀ .+ Vx)./2.0
    V̄y = (Vy₀ .+ Vy)./2.0

    return V̄x, V̄y
        
end

"""
    surface_V(H, B, Δx, Δy, temp, sim, A_noise, θ=[], UA_f=[])

Computes the ice surface velocity for a given glacier state
"""
function surface_V!(H::Matrix{F}, simulation::SIM) where {F <: AbstractFloat, SIM <: Simulation}
    params::Parameters = simulation.parameters
    ft = params.simulation.float_type
    it = params.simulation.int_type
    iceflow_model = simulation.model.iceflow
    glacier::Glacier = simulation.glaciers[iceflow_model.glacier_idx[]]
    B::Matrix{ft} = glacier.B
    H̄::Matrix{F} = iceflow_model.H̄
    dSdx::Matrix{ft} = iceflow_model.dSdx
    dSdy::Matrix{ft} = iceflow_model.dSdy
    ∇S::Matrix{ft} = iceflow_model.∇S
    ∇Sx::Matrix{ft} = iceflow_model.∇Sx
    ∇Sy::Matrix{ft} = iceflow_model.∇Sy
    Γꜛ::Ref{ft} = iceflow_model.Γ
    D::Matrix{ft} = iceflow_model.D
    Dx::Matrix{ft} = iceflow_model.Dx
    Dy::Matrix{ft} = iceflow_model.Dy
    A::Ref{ft} = iceflow_model.A
    Δx::ft = glacier.Δx
    Δy::ft = glacier.Δy
    n::it = params.physical.n
    ρ::ft = params.physical.ρ
    g::ft = params.physical.g
    
    # Update glacier surface altimetry
    S::Matrix{ft} = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    diff_x!(dSdx, S, Δx)  
    diff_y!(dSdy, S, Δy) 
    avg_y!(∇Sx, dSdx)
    avg_x!(∇Sy, dSdy)
    ∇S .= (∇Sx.^2 .+ ∇Sy.^2).^((n - 1)/2) 

    avg!(H̄, H)
    Γꜛ[] = 2.0 * A[] * (ρ * g)^n / (n+1) # surface stress (not average)  # 1 / m^3 s 
    D .= Γꜛ[] .* H̄.^(n + 1) .* ∇S
    
    # Compute averaged surface velocities
    Vx = .-D .* ∇Sx
    Vy = .-D .* ∇Sy 

    return Vx, Vy    
end

"""
    surface_V(H, B, Δx, Δy, temp, sim, A_noise, θ=[], UA_f=[])

Computes the ice surface velocity for a given glacier state
"""
function surface_V(SIA2Dmodel, temp, sim, A_noise, θ=nothing, UA_f=nothing; testmode=false)
    
    B = SIA2Dmodel.B
    H = SIA2Dmodel.H
    Δx = SIA2Dmodel.Δx
    Δy = SIA2Dmodel.Δy
    
    # Update glacier surface altimetry
    S = B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇S = (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n[] - 1)/2) 
    
    @assert (sim == "UDE" || sim == "PDE" || sim == "UDE_inplace") "Wrong type of simulation. Needs to be 'UDE' , 'UDE_inplace' or 'PDE'."
    if (sim == "UDE" && !testmode) || (sim == "UDE_inplace" && !testmode)
        A = predict_A̅(UA_f, θ, [temp])[1] 
    elseif sim == "PDE" || testmode
        A = A_fake(temp, A_noise, noise)[1]
    end
    Γꜛ = 2.0 * A * (ρ[] * g[])^n[] / (n[]+1) # surface stress (not average)  # 1 / m^3 s 
    D = Γꜛ .* avg(H).^(n[] + 1) .* ∇S
    
    # Compute averaged surface velocities
    Vx = - D .* avg_y(dSdx)
    Vy = - D .* avg_x(dSdy)

    return Vx, Vy    
end