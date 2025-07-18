##########################################################
### Optional features to run ODINN with a system image ###
##########################################################
### To be used, please add PackageCompiler again in the Project.toml file and uncomment the lines below ###

# function __init__()
#     # Run this check when ODINN is loaded
#     check_sysimage()
#     @async build_system_image_async()  # Runs in the background
# end

# function check_sysimage()
#     if isfile(SYSIMAGE_PATH) && Base.JLOptions().image_file != SYSIMAGE_PATH
#         @warn """
#         A system image for ODINN exists but is not in use!
#         Restart Julia with:
        
#         julia --sysimage=$SYSIMAGE_PATH

#         This will greatly reduce startup time.
#         """
#     end
# end

# function build_system_image()
#     mkpath(SYSIMAGE_DIR)  # Ensure directory exists
#     @info "Building system image for ODINN. This may take several minutes..."
#     try
#         PackageCompiler.create_sysimage([:ODINN]; sysimage_path=SYSIMAGE_PATH)
#         @info "System image created successfully at $SYSIMAGE_PATH"
#     catch e
#         @warn "Failed to create system image: $e"
#     end
# end

# function build_system_image_async()
#     if isfile(SYSIMAGE_PATH)
#         @info "System image already exists. Skipping compilation."
#         return
#     end

#     @info "Building system image in a separate process..."
    
#     log_file = joinpath(SYSIMAGE_DIR, "sysimage_build.log")
#     script_file = joinpath(SYSIMAGE_DIR, "build_sysimage.jl")

#     # Write script to a file
#     write(script_file, """
#     import Pkg
#     Pkg.activate("$(Base.active_project())")
#     Pkg.instantiate()
#     using ODINN
#     ODINN.build_system_image()
#     """)

#     # Run the script in a separate process
#     julia_cmd = `julia --project=$(Base.active_project()) $script_file`

#     open(log_file, "w") do io
#         Base.run(pipeline(julia_cmd, stdout=io, stderr=io))
#     end

#     @info "System image build started. Check logs at $log_file."
# end

"""
    enable_multiprocessing(params::Sleipnir.Parameters) -> Int

Configures and enables multiprocessing based on the provided simulation parameters.

# Arguments
- `params::Sleipnir.Parameters`: A parameter object containing simulation settings, 
  including the number of workers (`params.simulation.workers`) and whether multiprocessing 
  is enabled (`params.simulation.multiprocessing`).

# Behavior
- If multiprocessing is enabled (`params.simulation.multiprocessing == true`) and the 
  specified number of workers (`params.simulation.workers`) is greater than 0:
  - Adds the required number of worker processes if the current number of processes 
    (`nprocs()`) is less than the specified number of workers.
  - Suppresses precompilation output on the worker processes and ensures the `ODINN` 
    module is loaded on all workers.
  - If the specified number of workers is 1, removes all worker processes.

# Returns
- The number of worker processes (`nworkers()`) after configuration.

# Notes
- This function uses `@eval` to dynamically add or remove worker processes.
- Precompilation output is suppressed on workers to reduce noise in the console.
"""
function enable_multiprocessing(params::Sleipnir.Parameters)
    procs = params.simulation.workers
    if procs > 0 && params.simulation.multiprocessing
        if parse(Bool, get(ENV, "ODINN_OVERWRITE_MULTI", "false"))
            @assert procs == nprocs() "In the documentation CI it is not possible to configure the number of workers for multiprocessing. It is hardcoded to $(nprocs()-1) in the yaml files but the one defined in the simulation parameters is workers=$(procs)."
        else
            if nprocs() < procs
                @eval begin
                    # if isfile(SYSIMAGE_PATH)
                    #     addprocs($procs - nprocs(); exeflags="--sysimage=$SYSIMAGE_PATH") # Use custom system image if available
                    # else
                    addprocs($procs - nprocs(); exeflags="--project") # Fallback to default if system image is missing
                    # end
                    println("Number of cores: ", nprocs())
                    println("Number of workers: ", nworkers())

                    # Suppress output only on workers by temporarily redirecting stdout and stderr
                    old_stdout = stdout
                    old_stderr = stderr

                    try
                        @info "[ODINN] $(nworkers()) workers precompiling... Please wait."
                        redirect_stdout(devnull)
                        redirect_stderr(devnull)
                        using Dates
                        @everywhere using Revise
                        @everywhere using ODINN
                    finally
                        redirect_stdout(old_stdout)
                        redirect_stderr(old_stderr)
                    end
                end
            elseif nprocs() != procs && procs == 1
                @eval begin
                rmprocs(workers(), waitfor=0)
                @info "Number of cores: $(nprocs())"
                @info "Number of workers: $(nworkers())"
                end # @eval
            end
        end
    end
    return nworkers()
end

function kill_julia_procs()
    @warn "Killing all Julia processes..."
    Base.run("pkill -9 julia")
end
