
function clean()
    atexit() do
        run(`$(Base.julia_cmd())`)
    end
    exit()
 end

"""
    Initialize_ODINN(processes, python_path)

Initializes ODINN by configuring multiprocessing for a given number of processes. 
"""
function enable_multiprocessing(params::Sleipnir.Parameters)
    procs = params.simulation.workers
    if procs > 0 && params.simulation.multiprocessing
        if nprocs() < procs
            @eval begin
            addprocs($procs - nprocs(); exeflags="--project")
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            @everywhere using ODINN
            end # @eval
        elseif nprocs() != procs && procs == 1
            @eval begin
            rmprocs(workers(), waitfor=0)
            println("Number of cores: ", nprocs())
            println("Number of workers: ", nworkers())
            end # @eval
        end
    end
    return nworkers()
end
