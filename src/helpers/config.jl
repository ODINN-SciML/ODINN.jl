
function __init__()
    if myid() == 1
        @info "Before importing ODINN, make sure you have configured PyCall and restarted the Julia session!"

        # Create structural folders if needed
        OGGM_path = joinpath(homedir(), "Python/OGGM_data")
        if !isdir(OGGM_path)
            mkpath(OGGM_path)
        end

        println("Initializing Python libraries...")
    end

    # Load Python packages
    copy!(netCDF4, pyimport("netCDF4"))
    copy!(cfg, pyimport("oggm.cfg"))
    copy!(utils, pyimport("oggm.utils"))
    copy!(workflow, pyimport("oggm.workflow"))
    copy!(tasks, pyimport("oggm.tasks"))
    copy!(graphics, pyimport("oggm.graphics"))
    copy!(bedtopo, pyimport("oggm.shop.bedtopo"))
    copy!(millan22, pyimport("oggm.shop.millan22"))
    copy!(MBsandbox, pyimport("MBsandbox.mbmod_daily_oneflowline"))
    copy!(salem, pyimport("salem"))
    copy!(pd, pyimport("pandas"))
    copy!(np, pyimport("numpy"))
    copy!(xr, pyimport("xarray"))
end

function clean()
    atexit() do
        run(`$(Base.julia_cmd())`)
    end
    exit()
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
