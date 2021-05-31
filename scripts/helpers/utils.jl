###   Utility functions for a wide variety of purposes  ###

###############################################
############  FUNCTIONS   #####################
###############################################

### Staggered grids ###

"""
    avg(A)

4-point average in a matrix with border padding
"""
function avg(A)
    println("A in: ", size(A))
    org_size = size(A)
    @views A .= 0.25 * ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )
    println("A after: ", size(A))
    A .= PaddedView(0.0, A, org_size, (2,2))
    println("A out: ", size(A))
end

"""
    avg_x(A)

2-point average on x-axis with border padding
"""
function avg_x(A)
    org_size = size(A)
    @views A .= 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )
    A .= PaddedView(0.0, A, org_size, (0,2))
end

"""
    avg_y(A)

2-point average on y-axis with border padding 
"""
function avg_y(A)
    println("A in: ", size(A))
    org_size = size(A)
    @views A .= 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )
    println("A after: ", size(A))
    A .= PaddedView(0.0, A, org_size, (0,2))
    println("A out: ", size(A))
end


"""
    inn(A)

Access inner matrix 
"""
@views inn(A) = A[2:end-1,2:end-1];


# function gradnorm(dAdx, dAdy)
#     org_size = size(A)
#     sqrt.(avg_y(dSdx).^2 .+ avg_x(dSdyy).^2)
# end

"""
    smooth!(A)

Smooth data contained in a matrix with one time step (CFL) of diffusion.
"""
@views function smooth!(A)
    A[2:end-1,2:end-1] .= A[2:end-1,2:end-1] .+ 1.0./4.1.*(diff(diff(A[:,2:end-1], dims=1), dims=1) .+ diff(diff(A[2:end-1,:], dims=2), dims=2))
    A[1,:]=A[2,:]; A[end,:]=A[end-1,:]; A[:,1]=A[:,2]; A[:,end]=A[:,end-1]
    return
end

"""
    void2nan!(x, void)

Convert empty matrix grid cells into NaNs
"""
function void2nan!(x, void)
    for i in eachindex(x)
        @inbounds x[i] = ifelse(x[i]==void, NaN, x[i])
    end
end

"""
    get_ELA(MB, DEM)

Returns the Equilibrium Line Altitude (ELA) from a glacier based on a mass balance matrix. 
"""
function get_ELA(MB, DEM)
    z_MB₀ = []
    for i in eachindex(DEM)
        println("MBᵢ: ", MB[i])
        if(abs(MB[i]) < 0.1)
            push!(z_MB₀, DEM[i])
        end
    end
    # Return average glacier altitude with MB = 0 as ELA
    println("ELA altitudes: ", z_MB₀)
    return mean(z_MB₀)
end

function get_annual_ELAs(MB, DEM)
    println("Getting annual ELAs...")
    annual_ELAs = []
    for year in 1:size(MB)[3]
        z_MB₀ = []
        for i in 1:size(DEM)[1]
            for j in 1:size(DEM)[2]
                if(abs(MB[i,j,year]) < 0.01)
                    push!(z_MB₀, DEM[i,j,year])
                end
            end
        end
        push!(annual_ELAs, mean(z_MB₀))
    end
    # Return average glacier altitude with MB = 0 as ELA
    return annual_ELAs
end

"""
    closest_index(x, val)

Return the index of the closest Array element
"""
function closest_index(x, val)
    ibest = eachindex(x)[begin]
    dxbest = abs(x[ibest]-val)
    for I in eachindex(x)
        dx = abs(x[I]-val)
        if dx < dxbest
            dxbest = dx
            ibest = I
        end
    end
    return ibest
    end 

"""
    buffer_mean(A, i)

Perform the mean of the last 5 elements of an Array
"""
function buffer_mean(A, i)
    A_buffer = zeros(size(A[:,:,1]))

    if(i-5 < 1)
        j = 1
    else
        j = i-5
    end
    
    for y in 1:size(A)[3]
        for n in 1:size(A)[1]
            for m in 1:size(A)[2]
                A_buffer[n,m] = mean(A[n,m,j:i])
            end
        end
    end

    #println("A_buffer: ", size(A_buffer))

    return A_buffer
end

"""
    timestep!(Δts, Δx, D, method)

Computes or updates the time step for temporal discretization of a differential equation
"""
function timestep!(Δts, Δx, D, method)
    # To do: include 'method' and Δt as parameters of the function. 
    # Right now, this runs just as before.
    #method = "explicit-adaptive"

    if method == "explicit-adaptive"
        # Compute the maximum diffusivity in order to pick a temporal step that garantees estability 
        D_max = maximum(D)
        Δt = η * ( Δx^2 / (2 * D_max ))
    elseif method == "explicit"
        ## add this as parameter
        Δt = 0.001
        D_max = maximum(D)
        if Δt / ( Δx^2 / (2 * D_max )) > 1 
            println("Stability condition is not satisfied\n")
        end
    elseif method == "implicit"
        println("Implicit method not yet implemented\n")
    end
    # TODO: uncomment once a solution to "Mutating arrays" problem is found
    #append!(Δts, Δt)
    #println("Δt: ", Δt)

    return Δt
end