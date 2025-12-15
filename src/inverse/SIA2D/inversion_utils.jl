### Utils for discrete adjoint

function diff_x_adjoint(I, Δx)
    O = zeros(Sleipnir.Float, (size(I, 1)+1, size(I, 2)))
    O[(begin + 1):end, :] += I
    O[1:(end - 1), :] -= I
    return O / Δx
end

function diff_y_adjoint(I, Δy)
    O = zeros(Sleipnir.Float, (size(I, 1), size(I, 2)+1))
    O[:, (begin + 1):end] += I
    O[:, 1:(end - 1)] -= I
    return O / Δy
end

function clamp_borders_dx(dS, H, η₀, Δx)
    return max.(min.(dS, η₀ * H[2:end, 2:(end - 1)]/Δx), -η₀ *
                                                         H[1:(end - 1), 2:(end - 1)]/Δx)
end

function clamp_borders_dx_adjoint!(∂dS, ∂H, ∂C, η₀, Δx, H, dS)
    # Note: this implementation doesn't hold if H is negative
    ∂dS .= ∂C .* ((dS .< η₀ * H[2:end, 2:(end - 1)]/Δx) .&
            (dS .> -η₀ * H[1:(end - 1), 2:(end - 1)]/Δx))
    ∂H[1:(end - 1), 2:(end - 1)] .= - (η₀ * ∂C / Δx) .*
                                    (dS .< -η₀ * H[1:(end - 1), 2:(end - 1)]/Δx)
    ∂H[2:end, 2:(end - 1)] += (η₀ * ∂C / Δx) .* (dS .> η₀ * H[2:end, 2:(end - 1)]/Δx)
end

function clamp_borders_dy(dS, H, η₀, Δy)
    return max.(min.(dS, η₀ * H[2:(end - 1), 2:end]/Δy), -η₀ *
                                                         H[2:(end - 1), 1:(end - 1)]/Δy)
end

function clamp_borders_dy_adjoint!(∂dS, ∂H, ∂C, η₀, Δy, H, dS)
    # Note: this implementation doesn't hold if H is negative
    ∂dS .= ∂C .* ((dS .< η₀ * H[2:(end - 1), 2:end]/Δy) .&
            (dS .> -η₀ * H[2:(end - 1), 1:(end - 1)]/Δy))
    ∂H[2:(end - 1), 1:(end - 1)] .= - (η₀ * ∂C / Δy) .*
                                    (dS .< -η₀ * H[2:(end - 1), 1:(end - 1)]/Δy)
    ∂H[2:(end - 1), 2:end] += (η₀ * ∂C / Δy) .* (dS .> η₀ * H[2:(end - 1), 2:end]/Δy)
end

function avg_adjoint(I)
    O = zeros(Sleipnir.Float, (size(I, 1)+1, size(I, 2)+1))
    O[1:(end - 1), 1:(end - 1)] += I
    O[2:end, 1:(end - 1)] += I
    O[1:(end - 1), 2:end] += I
    O[2:end, 2:end] += I
    return 0.25*O
end

function avg_x_adjoint(I)
    O = zeros(Sleipnir.Float, (size(I, 1)+1, size(I, 2)))
    O[1:(end - 1), :] += I
    O[2:end, :] += I
    return 0.5*O
end

function avg_y_adjoint(I)
    O = zeros(Sleipnir.Float, (size(I, 1), size(I, 2)+1))
    O[:, 1:(end - 1)] += I
    O[:, 2:end] += I
    return 0.5*O
end
