
function __init__()

    # Create structural folders if needed
    OGGM_path = joinpath(homedir(), "Python/OGGM_data")
    if !isdir(OGGM_path)
        mkpath(OGGM_path)
    end

    # Load Python packages
    println("Initializing Python libraries...")
    copy!(netCDF4, pyimport_conda("netCDF4", "netCDF4"))
    copy!(cfg, pyimport_conda("oggm.cfg", "oggm"))
    copy!(utils, pyimport_conda("oggm.utils", "oggm"))
    copy!(workflow, pyimport_conda("oggm.workflow", "oggm"))
    copy!(tasks, pyimport_conda("oggm.tasks", "oggm"))
    copy!(graphics, pyimport_conda("oggm.graphics", "oggm"))
    copy!(bedtopo, pyimport_conda("oggm.shop.bedtopo", "oggm"))
    copy!(MBsandbox, pyimport_conda("MBsandbox.mbmod_daily_oneflowline", "MBsandbox"))
    copy!(np, pyimport_conda("numpy", "numpy"))
    copy!(xr, pyimport_conda("xarray", "xarray"))
end

"""
    Initialize_ODINN(processes, python_path)

Initializes ODINN by configuring PyCall based on a given Python path. It also configures multiprocessing
for a given number of processes. 
"""
function enable_multiprocessing(procs)

    if procs > 1
        if nprocs() < procs
            global processes = procs
            @eval begin
            addprocs(processes - nprocs(); exeflags="--project")
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            @everywhere using ODINN
            end # @eval
        end
    end

end