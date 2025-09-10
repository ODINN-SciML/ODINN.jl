"""
    random_matrix(mean, std, corr_length)

Generate a random matrix with entries drawn from a multivariate normal distribution
whose mean is given by `mean` and whose covariance decays exponentially with
grid distance.

# Arguments
- `mean::AbstractMatrix{<:Real}`: Matrix specifying the spatial mean values at each grid point.
    Entries equal to `0.0` are treated as inactive and skipped in sampling.
- `std::Real`: Standard deviation scaling factor for the covariance kernel.
- `corr_length::Real`: Correlation length parameter controlling how fast correlations
    decay with Euclidean distance between grid points.

# Returns
- `H_sample::Matrix{Float64}`: A random realization of the same size as `mean`, with correlated entries 
    drawn from `MvNormal(mean, Σ)`, where
    `Σ[i,j] = std * exp(-‖coords[i] - coords[j]‖ / corr_length)`.
"""
function random_matrix(mean, std, corr_length)
    nx, ny = size(mean)
    N = nx * ny
    # Sample coordinates
    μ = [mean[i, j] for i in 1:nx for j in 1:ny]
    coords = [(i,j) for i in 1:nx for j in 1:ny]
    Σ = zeros(N, N)
    for i in 1:N
        for j in 1:N
            d = norm(coords[i] .- coords[j])
            Σ[i, j] = std * exp(- d / corr_length)
        end
    end
    # Sample from multivariate normal
    mvnorm = MvNormal(μ, Σ)
    sample = rand(mvnorm)
    H_sample = zeros(nx, ny)
    for (k, (i, j)) in enumerate(coords)
        if mean[i, j] == 0.0
            continue
        end
        H_sample[i, j] = sample[k]
    end
    return H_sample
end