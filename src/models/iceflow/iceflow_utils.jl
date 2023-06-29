
# Helper functions for the staggered grid

####  Non-allocating functions  ####

diff_x!(O, I, Δx) = @views @. O = (I[begin+1:end,:] - I[1:end-1,:]) / Δx

diff_y!(O, I, Δy) = @views @. O = (I[:,begin+1:end] - I[:,1:end - 1]) / Δy

avg!(O, I) = @views @. O = (I[1:end-1,1:end-1] + I[2:end,1:end-1] + I[1:end-1,2:end] + I[2:end,2:end]) * 0.25

avg_x!(O, I) = @views @. O = (I[1:end-1,:] + I[2:end,:]) * 0.5

avg_y!(O, I) = @views @. O = (I[:,1:end-1] + I[:,2:end]) * 0.5

####  Allocating functions  ####

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

"""
    inn1(A)

Access the inner part of the matrix (-1,-1)
"""
@views inn1(A) = A[1:end-1,1:end-1]