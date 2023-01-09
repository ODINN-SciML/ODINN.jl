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
        # println("Using $ice_thickness_source for initial state")
        # Retrieve initial conditions from OGGM
        # initial ice thickness conditions for forward model
        if ice_thickness_source == "millan"
            H₀ = Float64.(ifelse.(glacier_gd.glacier_mask.data .== 1, glacier_gd.millan_ice_thickness.data, 0.0))
        elseif ice_thickness_source == "farinotti"
            H₀ = Float64.(glacier_gd.consensus_ice_thickness.data)
        end
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
        # println("Using spin-up for initial state")
        # Retrieve initial state from previous spinup simulation
        gdir_spinup = load(joinpath(ODINN.root_dir, "data/spinup/gdir_refs.jld2"))["gdir_refs"]
        H₀ = similar(gdir_spinup[1]["H"])
        found = false
        for i in 1:length(gdir_spinup)
            if gdir_spinup[i]["RGI_ID"] == gdir.rgi_id
                H₀ = gdir_spinup[i]["H"]
                found = true
                break
            end
        end

        @assert found == true "Spin up glacier simulation not found for $(gdir.rgi_id)."

    end
    try
        H = deepcopy(H₀)
        B = glacier_gd.topo.data .- H₀ # bedrock
        S = Float64.(glacier_gd.topo.data) # surface elevation
        V = Float64.(ifelse.(glacier_gd.glacier_mask.data .== 1, glacier_gd.millan_v.data, 0.0))
        fillNaN!(V)
        nx = glacier_gd.y.size # glacier extent
        ny = glacier_gd.x.size # really weird, but this is inversed 
        Δx = abs(gdir.grid.dx)
        Δy = abs(gdir.grid.dy)

        return H₀, H, S, B, V, (nx,ny), (Δx,Δy)
    catch
        missing_glaciers = load(joinpath(ODINN.root_dir, "data/missing_glaciers.jld2"))["missing_glaciers"]
        push!(missing_glaciers, gdir.rgi_id)
        jldsave(joinpath(ODINN.root_dir, "data/missing_glaciers.jld2"); missing_glaciers)
        @warn "Glacier without data: $(gdir.rgi_id). Updating list of missing glaciers. Please try again."
    end

    glacier_gd.close() # Release any resources linked to this object

end

function build_PDE_context(gdir, longterm_temp, climate_years, A_noise, tspan; run_spinup=false, random_MB=nothing)
    # Determine initial geometry conditions
    H₀, H, S, B, V, nxy, Δxy = get_initial_geometry(gdir, run_spinup)
    rgi_id = gdir.rgi_id
    # Initialize all matrices for the solver
    nx, ny = nxy
    S, dSdx, dSdy = zeros(Float64,nx,ny),zeros(Float64,nx-1,ny),zeros(Float64,nx,ny-1)
    dSdx_edges, dSdy_edges, ∇S = zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1),zeros(Float64,nx-1,ny-1)
    D, Fx, Fy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-2),zeros(Float64,nx-2,ny-1)
    V, Vx, Vy = zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1),zeros(Float64,nx-1,ny-1)
    MB = zeros(Float64,nx,ny)
    A = [2e-16]
    α = [0.0]                      # Weertman-type basal sliding (Weertman, 1964, 1972). 1 -> sliding / 0 -> no sliding
    C = [15e-14]    
    Γ = [0.0]
    maxS, minS = [0.0], [0.0]     
    simulation_years = collect(Int(tspan[1]):Int(tspan[2]))
    
    # Gather simulation parameters
    current_year = [0] 
    if isnothing(random_MB)
        random_MB = zeros(Float64,Int(tspan[2]) - Int(tspan[1]))
    end
    context = ArrayPartition(A, B, S, dSdx, dSdy, D, longterm_temp, dSdx_edges, dSdy_edges, ∇S, Fx, Fy, Vx, Vy, V, C, α, 
                            current_year, nxy, Δxy, H₀, tspan, A_noise, random_MB, MB, rgi_id, Γ, maxS, minS, climate_years, simulation_years)
    return context, H
end

function build_UDE_context(gdir, climate_years, tspan; run_spinup=false, random_MB=nothing)
    H₀, H, S, B, V, nxy, Δxy = get_initial_geometry(gdir, run_spinup)
    simulation_years = collect(tspan[1]:tspan[2])

    context = (B, H₀, H, nxy, Δxy, tspan, random_MB, gdir.rgi_id, S, V, climate_years, simulation_years)

    return context
end

# UDE  context using Glathida for H
function build_UDE_context_inv(gdir, gdir_ref, tspan;  run_spinup=false)
    H₀, H₁, S, B, V, nxy, Δxy = get_initial_geometry(gdir, run_spinup)
    rgi_id = gdir.rgi_id
    # Get evolved tickness and surface
    H = gdir_ref["H"]

    @ignore begin
        glacier_gd = xr.open_dataset(gdir.get_filepath("gridded_data"))
        H₁ = glacier_gd.consensus_ice_thickness.data
        fillNaN!(H₀) # Fill NaNs with 0s to have real boundary conditions
        # smooth!(H₁)
        heatmap_diff = Plots.heatmap(H₀ .- H, title="Spin-up vs Reference difference")
        heatmap_diff2 = Plots.heatmap(H₁ .- H₀, title="Farinotti vs spin-up difference")
        heatmap_diff3 = Plots.heatmap(H₁ .- H, title="Farinotti vs Reference difference")
        training_path = joinpath(root_plots,"inversions")
        Plots.savefig(heatmap_diff, joinpath(training_path, "H_diff_su_ref.pdf"))
        Plots.savefig(heatmap_diff2, joinpath(training_path, "H_diff_far_spu.pdf"))
        Plots.savefig(heatmap_diff3, joinpath(training_path, "H_diff_far_ref.pdf"))
    end

    context = (nxy, Δxy, tspan, rgi_id, S, V, H₀)

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

function get_glathida_path_and_IDs()
    # Download all data from Glathida
    # TODO: make Download work with specific path
    gtd_file = Downloads.download("https://cluster.klima.uni-bremen.de/~oggm/glathida/glathida-v3.1.0/data/TTT_per_rgi_id.h5")
    # gtd_file = Downloads.download("https://cluster.klima.uni-bremen.de/~oggm/glathida/glathida-v3.1.0/data/TTT_per_rgi_id.h5", gtd_path)

    glathida = pd.HDFStore(gtd_file)
    rgi_ids = glathida.keys()
    rgi_ids = [id[2:end] for id in rgi_ids]

    # glathida = h5open(gtd_path, "r")
    # # Retrieve RGI IDs with Glathida data
    # rgi_ids = keys(glathida)
    # Delete Greenland and Antarctic glaciers, for now
    # deleteat!(rgi_ids, findall(x->x[begin:8]=="RGI60-05",rgi_ids))
    # deleteat!(rgi_ids, findall(x->x[begin:8]=="RGI60-19",rgi_ids))

    # Delete missing glaciers in Glathida from the Millan 22 open_dataset
    # for missing_glacier in missing_glaciers
    #     deleteat!(rgi_ids, findall(x->x==missing_glacier,rgi_ids))  
    # end

    return gtd_file, rgi_ids
end

"""
    get_glathida(gdirs)

Downloads and creates distributed ice thickness matrices for each gdir based on the Glathida observations.
"""
function get_glathida!(gtd_file, gdirs; force=false)
    glathida = pd.HDFStore(gtd_file)
    # TODO: make this work in a pmap
    gtd_grids = map(gdir -> get_glathida_glacier(gdir, glathida, force), gdirs) 

    # Update missing_glaciers list before removing them
    missing_glaciers = load(joinpath(ODINN.root_dir, "data/missing_glaciers.jld2"))["missing_glaciers"]
    for (gtd_grid, gdir) in zip(gtd_grids, gdirs)
        if (length(gtd_grid[gtd_grid .!= 0.0]) == 0) && all(gdir.rgi_id .!= missing_glaciers)
            push!(missing_glaciers, gdir.rgi_id)
            @info "Glacier with all data at 0: $(gdir.rgi_id). Updating list of missing glaciers..."
        end
    end
    jldsave(joinpath(ODINN.root_dir, "data/missing_glaciers.jld2"); missing_glaciers)

    # Remove glaciers with all data points at 0
    deleteat!(gtd_grids, findall(x->length(x[x .!= 0.0])==0, gtd_grids))
    deleteat!(gdirs, findall(x->length(x[x .!= 0.0])==0, gtd_grids))
    return gtd_grids
end

"""
    get_glathida(gdir, glathida)

Either retrieves computes and writes the Glathida ice thickness matrix for a given gdir.
"""
function get_glathida_glacier(gdir, glathida, force)
    gtd_path = joinpath(gdir.dir, "glathida.h5")
    if isfile(gtd_path) && !force
        gtd_grid = h5read(gtd_path, "gtd_grid")
    else
        df_gtd = glathida[gdir.rgi_id]
        jj, ii = gdir.grid.transform(df_gtd["POINT_LON"], df_gtd["POINT_LAT"], crs=salem.wgs84, nearest=true)

        gtd_grid = zeros((gdir.grid.ny,gdir.grid.nx))
        for (thick, i, j) in zip(df_gtd["THICKNESS"], ii, jj)
            if gtd_grid[i,j] != 0.0
                gtd_grid[i,j] = (gtd_grid[i,j] + thick)/2.0 # average
            else
                gtd_grid[i,j] = thick
            end
        end
        # Save file 
        h5open(joinpath(gdir.dir, "glathida.h5"), "w") do file
            write(file, "gtd_grid", gtd_grid)  
        end
    end

    return gtd_grid
end

function filter_missing_glaciers!(gdirs::Vector{PyObject})
    task_log = global_tasks.compile_task_log(gdirs, 
                                            task_names=["gridded_attributes", "velocity_to_gdir", "thickness_to_gdir"])
                                                        
    task_log.to_csv(joinpath(ODINN.root_dir, "task_log.csv"))
    glacier_filter = ((task_log.velocity_to_gdir != "SUCCESS").values .&& (task_log.gridded_attributes != "SUCCESS").values
                        .&& (task_log.thickness_to_gdir != "SUCCESS").values)
    glacier_ids = []
    for id in task_log.index
        push!(glacier_ids, id)
    end
    missing_glaciers = glacier_ids[glacier_filter]

    try
        missing_glaciers_old = load(joinpath(ODINN.root_dir, "data/missing_glaciers.jld2"))["missing_glaciers"]
        for missing_glacier in missing_glaciers_old
            if all(missing_glacier .!= missing_glaciers) # if the glacier is already not present, let's add it
                push!(missing_glaciers, missing_glacier)
            end
        end
    catch error
        @warn "$error: No missing_glaciers.jld file available. Skipping..."
    end

    for id in missing_glaciers
        deleteat!(gdirs, findall(x->x.rgi_id==id, gdirs))
    end
    # Save missing glaciers in a file
    jldsave(joinpath(ODINN.root_dir, "data/missing_glaciers.jld2"); missing_glaciers)
    # @warn "Filtering out these glaciers from gdir list: $missing_glaciers"
    
    return missing_glaciers
end

function filter_missing_glaciers!(rgi_ids::Vector{String})


    # Check which glaciers we can actually process
    rgi_stats = pd.read_csv(utils.file_downloader("https://cluster.klima.uni-bremen.de/~oggm/rgi/rgi62_stats.csv"), index_col=0)
    # rgi_stats = rgi_stats.loc[rgi_ids]

    # if any(rgi_stats.Connect .== 2)
    #     @warn "You have some level 2 glaciers... Removing..."
    #     rgi_ids = [rgi_stats.loc[rgi_stats.Connect .!= 2].index]
    # end

    indices = [rgi_stats.index...]
    for rgi_id in rgi_ids
        if rgi_stats.Connect.values[indices .== rgi_id] == 2
            @warn "Filtering glacier $rgi_id..."
            deleteat!(rgi_ids, rgi_ids .== rgi_id)
        end

    end

    try
        missing_glaciers = load(joinpath(ODINN.root_dir, "data/missing_glaciers.jld2"))["missing_glaciers"]
        for missing_glacier in missing_glaciers
            deleteat!(rgi_ids, findall(x->x == missing_glacier,rgi_ids))
        end
        @info "Filtering out these glaciers from RGI ID list: $missing_glaciers"
    catch error
        @warn "$error: No missing_glaciers.jld file available. Skipping..."
    end
    
end
