export generate_random_MB

###############################################
############  FUNCTIONS   #####################
###############################################

function generate_random_MB(gdirs_climate, tspan)
    random_MB = [] # tuple with RGI_ID, MB_max, MB_min
    for (gdir, climate) in zip(gdirs_climate[2], gdirs_climate[3])
        clim = 1/abs(mean(climate)) # climate driver to adapt random MB
        clim = (clim <= 1.3)*clim # clip large values
        MB_max = (ref_max_MB .+ 3 .* randn(MersenneTwister(1),floor(Int,tspan[2]))).*clim
        MB_min = (ref_min_MB .+ 3 .* randn(MersenneTwister(2),floor(Int,tspan[2]))).*clim
        push!(random_MB, (gdir.rgi_id, MB_max, MB_min))
    end

    return random_MB
end