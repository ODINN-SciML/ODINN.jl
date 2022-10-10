# Helper functions for the staggered grid
"""
    avg(A)

4-point average of a matrix
"""
@views avg(A) = 0.25f0 .* ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )

"""
    avg_x(A)

2-point average of a matrix's X axis
"""
@views avg_x(A) = 0.5f0 .* ( A[1:end-1,:] .+ A[2:end,:] )

"""
    avg_y(A)

2-point average of a matrix's Y axis
"""
@views avg_y(A) = 0.5f0 .* ( A[:,1:end-1] .+ A[:,2:end] )

"""
    diff_x(A)

2-point differential of a matrix's X axis
"""
@views diff_x(A) = (A[begin + 1:end, :] .- A[1:end - 1, :])

"""
    diff_y(A)

2-point differential of a matrix's Y axis
"""
@views diff_y(A) = (A[:, begin + 1:end] .- A[:, 1:end - 1])

"""
    inn(A)

Access the inner part of the matrix (-2,-2)
"""
@views inn(A) = A[2:end-1,2:end-1]

"""
fillNaN!(x, fill)

Convert empty matrix grid cells into fill value
"""
function fillNaN!(A, fill=zero(eltype(A)))
    for i in eachindex(A)
        @inbounds A[i] = ifelse(isnan(A[i]), fill, A[i])
    end
end

function fillNaN(A, fill=zero(eltype(A)))
    return @. ifelse(isnan(A), fill, A)
end

function fillZeros!(A, fill=NaN)
    for i in eachindex(A)
        @inbounds A[i] = ifelse(iszero(A[i]), fill, A[i])
    end
end

function fillZeros(A, fill=NaN)
    return @. ifelse(iszero(A), fill, A)
end

"""
    smooth!(A)

Smooth data contained in a matrix with one time step (CFL) of diffusion.
"""
@views function smooth!(A)
    A[2:end-1,2:end-1] .= A[2:end-1,2:end-1] .+ 1.0f0./4.1f0.*(diff(diff(A[:,2:end-1], dims=1), dims=1) .+ diff(diff(A[2:end-1,:], dims=2), dims=2))
    A[1,:]=A[2,:]; A[end,:]=A[end-1,:]; A[:,1]=A[:,2]; A[:,end]=A[:,end-1]
    return
end

function smooth(A)
    A_smooth = A[2:end-1,2:end-1] .+ 1.0f0./4.1f0.*(diff(diff(A[:,2:end-1], dims=1), dims=1) .+ diff(diff(A[2:end-1,:], dims=2), dims=2))
    @tullio A_smooth_pad[i,j] := A_smooth[pad(i-1,1,1),pad(j-1,1,1)] # Fill borders 
    return A_smooth_pad
end

function reset_epochs()
    @everywhere @eval ODINN global current_epoch = 1
end

function set_current_epoch(epoch)
    @everywhere @eval ODINN global current_epoch = $epoch
end

function make_plots(plots_i)
    @everywhere @eval ODINN global plots = $plots_i
end

function set_use_MB(use_MB_i)
    @everywhere @eval ODINN global use_MB = $use_MB_i
end

function set_run_spinup(run_spinup_i)
    @everywhere @eval ODINN global run_spinup = $run_spinup_i
end

function set_use_spinup(use_spinup_i)
    @everywhere @eval ODINN global use_spinup = $use_spinup_i
end

function set_create_ref_dataset(create_ref_dataset_i)
    @everywhere @eval ODINN global create_ref_dataset = $create_ref_dataset_i
end

function set_train(train_i)
    @everywhere @eval ODINN global train = $train_i
end

function set_retrain(retrain_i)
    @everywhere @eval ODINN global retrain = $retrain_i
end

function get_gdir_refs(refs, gdirs)
    gdir_refs = []
    for (ref, gdir) in zip(refs, gdirs)
        push!(gdir_refs, Dict("RGI_ID"=>gdir.rgi_id,
                                "H"=>ref["H"],
                                "Vx"=>ref["Vx"],
                                "Vy"=>ref["Vy"],
                                "S"=>ref["S"],
                                "B"=>ref["B"]))
    end
    return gdir_refs
end
