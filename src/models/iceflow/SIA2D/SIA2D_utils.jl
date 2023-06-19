

"""
SIA!(dH, H, SIA2Dmodel)

Compute an in-place step of the Shallow Ice Approximation PDE in a forward model
"""
function SIA!(dH, H, SIA2Dmodel)
    # Retrieve parameters
    A::Float64 = SIA2Dmodel.A
    B::Matrix{Float64} = SIA2Dmodel.B
    S::Matrix{Float64} = SIA2Dmodel.S
    dSdx::Matrix{Float64} = SIA2Dmodel.dSdx
    dSdy::Matrix{Float64} = SIA2Dmodel.dSdy
    D::Matrix{Float64} = SIA2Dmodel.D
    dSdx_edges::Matrix{Float64} = SIA2Dmodel.dSdx_edges
    dSdy_edges::Matrix{Float64} = SIA2Dmodel.dSdy_edges
    ∇S::Matrix{Float64} = SIA2Dmodel.∇S
    Fx::Matrix{Float64} = SIA2Dmodel.Fx
    Fy::Matrix{Float64} = SIA2Dmodel.Fy
    Δx::Float64 = SIA2Dmodel.Δx
    Δy::Float64 = SIA2Dmodel.Δy
    Γ::Float64 = SIA2Dmodel.Γ

    # Update glacier surface altimetry
    S .= B .+ H

    # All grid variables computed in a staggered grid
    # Compute surface gradients on edges
    diff_x!(dSdx, S, Δx)  
    diff_y!(dSdy, S, Δy) 
    ∇S .= (avg_y(dSdx).^2 .+ avg_x(dSdy).^2).^((n - 1)/2) 

    Γ = 2.0 * A * (ρ * g)^n / (n+2) # 1 / m^3 s 
    D .= Γ .* avg(H).^(n + 2) .* ∇S

    # Compute flux components
    @views diff_x!(dSdx_edges, S[:,2:end - 1], Δx)
    @views diff_y!(dSdy_edges, S[2:end - 1,:], Δy)
    # Cap surface elevaton differences with the upstream ice thickness to 
    # imporse boundary condition of the SIA equation
    η₀ = 1.0
    dSdx_edges .= min.(dSdx_edges,  η₀ * H[1:end-1, 2:end-1]./Δx,  η₀ * H[2:end, 2:end-1]./Δx)
    dSdy_edges .= min.(dSdy_edges,  η₀ * H[2:end-1, 1:end-1]./Δy,  η₀ * H[2:end-1, 2:end]./Δy) 
    dSdx_edges .= max.(dSdx_edges, -η₀ * H[1:end-1, 2:end-1]./Δx, -η₀ * H[2:end, 2:end-1]./Δx)
    dSdy_edges .= max.(dSdy_edges, -η₀ * H[2:end-1, 1:end-1]./Δy, -η₀ * H[2:end-1, 2:end]./Δy)

    Fx .= .-avg_y(D) .* dSdx_edges
    Fy .= .-avg_x(D) .* dSdy_edges 

    #  Flux divergence
    inn(dH) .= .-(diff_x(Fx) ./ Δx .+ diff_y(Fy) ./ Δy) 

end

"""
    SIA(H, A, context)

Compute a step of the Shallow Ice Approximation UDE in a forward model. Allocates memory.
"""
function SIA(H, SIA2Dmodel)
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
function avg_surface_V(SIA2Dmodel, temp, sim, θ=nothing, UA_f=nothing; testmode=false)
    # TODO: see how to get this
    # A_noise = parameters.A_noise
    # Δx, Δy, A_noise = retrieve_context(context, sim)

    # We compute the initial and final surface velocity and average them
    # TODO: Add more datapoints to better interpolate this
    Vx₀, Vy₀ = surface_V(SIA2Dmodel, sim, A_noise, θ, UA_f; testmode=testmode)
    Vx, Vy = surface_V(SIA2Dmodel, temp, sim, A_noise, θ, UA_f; testmode=testmode)
    V̄x = (Vx₀ .+ Vx)./2.0
    V̄y = (Vy₀ .+ Vy)./2.0

    return V̄x, V̄y
        
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