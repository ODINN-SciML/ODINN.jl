###############################################
############  FUNCTIONS   #####################
###############################################

"""
    get_glacier_data(glacier_gd)

Retrieves the initial glacier geometry (bedrock + ice thickness) for a glacier with other necessary data (e.g. grid size and ice surface velocities).
"""
function get_initial_geometry(gdir, run_spinup, smoothing=true)
    # Load glacier gridded data
    glacier_gd = xr.open_dataset(gdir.get_filepath("gridded_data"))
    if run_spinup || !use_spinup[]
        # Retrieve initial conditions from OGGM
        H₀ = Float32.(glacier_gd.consensus_ice_thickness.data) # initial ice thickness conditions for forward model
        fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
        if smoothing 
            smooth!(H₀)  # Smooth initial ice thickness to help the solver
        end

        # Create path for spinup simulation results
        gdir_path =  dirname(gdir.get_filepath("dem"))
        if !isdir(gdir_path)
            mkdir(gdir_path)
        end
    else
        # Retrieve initial state from previous spinup simulation
        gdir_refs = load(joinpath(ODINN.root_dir, "data/gdir_refs.jld2"))["gdir_refs"]
        H₀ = similar(gdir_refs[1]["H"])
        found = false
        for i in 1:length(gdir_refs)
            if gdir_refs[i]["RGI_ID"] == gdir.rgi_id
                H₀ = gdir_refs[i]["H"]
                found = true
                break
            end
        end
        @assert found == true "Spin up glacier simulation not found for $(gdir.rgi_id)."

    end

    H = deepcopy(H₀)
    B = glacier_gd.topo.data .- H₀ # bedrock
    S = glacier_gd.topo.data # surface elevation
    V = glacier_gd.millan_v.data
    nx = glacier_gd.y.size # glacier extent
    ny = glacier_gd.x.size # really weird, but this is inversed 
    Δx = Float32(abs(gdir.grid.dx))
    Δy = Float32(abs(gdir.grid.dy))

    return H₀, H, S, B, V, (nx,ny), (Δx,Δy)

end

function build_PDE_context(gdir, longterm_temp, A_noise, tspan; run_spinup=false, random_MB=nothing)
    # Determine initial geometry conditions
    H₀, H, S, B, V, nxy, Δxy = get_initial_geometry(gdir, run_spinup)
    rgi_id = gdir.rgi_id
    # Initialize all matrices for the solver
    nx, ny = nxy
    S, dSdx, dSdy = zeros(Float32,nx,ny),zeros(Float32,nx-1,ny),zeros(Float32,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1),zeros(Float32,nx-1,ny-1)
    D, Fx, Fy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-2),zeros(Float32,nx-2,ny-1)
    V, Vx, Vy = zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1),zeros(Float32,nx-1,ny-1)
    MB = zeros(Float32,nx-2,ny-2)
    A = [2f-16]
    α = [0.0f0]                      # Weertman-type basal sliding (Weertman, 1964, 1972). 1 -> sliding / 0 -> no sliding
    C = [15f-14]    
    Γ = [0.0f0]
    maxS, minS = [0.0f0], [0.0f0]     
    
    # Gather simulation parameters
    current_year = [0.0f0] 
    if isnothing(random_MB)
        random_MB = zeros(Float32,Int(tspan[2]))
    end
    context = ArrayPartition(A, B, S, dSdx, dSdy, D, longterm_temp, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, 
                            current_year, nxy, Δxy, H₀, tspan, A_noise, random_MB, MB, rgi_id, Γ, maxS, minS)
    return context, H
end

function build_UDE_context(gdir, tspan; run_spinup=false, random_MB=nothing)
    H₀, H, S, B, V, nxy, Δxy = get_initial_geometry(gdir, run_spinup)
    rgi_id = gdir.rgi_id

    # Tuple with all the temp series
    context = (B, H₀, H, nxy, Δxy, tspan, random_MB, rgi_id, S, V)

    return context
end

# UDE  context using Glathida for H
function build_UDE_context(gdir, glathida, tspan)
    H₀, H, S, B, V, nxy, Δxy = get_initial_geometry(gdir, run_spinup)
    rgi_id = gdir.rgi_id

    # Tuple with all the temp series
    context = (B, H₀, H, nxy, Δxy, tspan, random_MB, rgi_id, S, V)

    return context
end

"""
    retrieve_context(context::Tuple)

Retrieves context variables for computing the surface velocities of the UDE.
"""
function retrieve_context(context::Tuple)
    # context = (B, H₀, H, Vxy_obs, nxy, Δxy, tspan)
    B = context[1]
    H₀ = context[2]
    Δx = context[5][1]
    Δy = context[5][2]
    return B, H₀, Δx, Δy, nothing
end

"""
    retrieve_context(context::ArrayPartition)

Retrieves context variables for computing the surface velocities of the PDE.
"""
function retrieve_context(context::ArrayPartition)
    # context = ArrayPartition([A], B, S, dSdx, dSdy, D, longterm_temp, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, [current_year], nxy, Δxy, H₀, tspan)
    B = context.x[2]
    H₀ = context.x[21]
    Δx = context.x[20][1]
    Δy = context.x[20][2]
    A_noise = context.x[23]
    return B, H₀, Δx, Δy, A_noise
end

