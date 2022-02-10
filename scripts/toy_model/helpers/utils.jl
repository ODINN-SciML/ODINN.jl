
# Helper functions for the staggered grid
"""
    avg(A)

4-point average of a matrix
"""
@views avg(A) = 0.25 .* ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )

"""
    avg_x(A)

2-point average of a matrix's X axis
"""
@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )

"""
    avg_y(A)

2-point average of a matrix's Y axis
"""
@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )

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
