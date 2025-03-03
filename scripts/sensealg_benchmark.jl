using Pkg; Pkg.activate(".")

using Revise
using Enzyme

# Enzyme.API.runtimeActivity!(true)
# Enzyme.API.printall!(true)
# Enzyme.API.instname!(true)

using ODINN
using Test

rgi_paths = get_rgi_paths()

working_dir = joinpath(homedir(), "OGGM/ODINN_tests")

# sensealgs = [ODINN.InterpolatingAdjoint(autojacvec=ODINN.ReverseDiffVJP()),
#             ODINN.InterpolatingAdjoint(autojacvec=ODINN.ZygoteVJP()), 
#             ODINN.InterpolatingAdjoint(autojacvec=ODINN.EnzymeVJP()),
#             ODINN.InterpolatingAdjoint(autojacvec=nothing),
#             ODINN.QuadratureAdjoint(autojacvec=ODINN.ReverseDiffVJP()), 
#             ODINN.QuadratureAdjoint(autojacvec=ODINN.ZygoteVJP()), 
#             ODINN.QuadratureAdjoint(autojacvec=ODINN.EnzymeVJP()), 
#             ODINN.QuadratureAdjoint(autojacvec=nothing), 
#             ODINN.BacksolveAdjoint(autojacvec=ODINN.ReverseDiffVJP()), 
#             ODINN.BacksolveAdjoint(autojacvec=ODINN.ZygoteVJP())]

# adtypes = [ODINN.AutoZygote(), 
#            ODINN.AutoReverseDiff(compile=false), 
#            ODINN.AutoEnzyme()]

# autojacvec ReverseDiffVJP returns 
# MethodError: no method matching SIA2D_UDE(::ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64}, Matrix{Float64}}, ::ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}, ::ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}, ::FunctionalInversion, ::Int64)
# which is a type conversion problem when computing the adjoint using reverse diff

# Just EnzymeVJP and ReverseDiffVJP are compatible with Interpolating adjoints

# sensealgs = [ODINN.ZygoteAdjoint()]
# adtypes   = [ODINN.NoAD()]

sensealgs = [ODINN.QuadratureAdjoint(autojacvec=ODINN.ReverseDiffVJP(true)), 
             ODINN.GaussAdjoint(autojacvec=ODINN.EnzymeVJP())]
adtypes = [ODINN.AutoZygote()]

@testset "Run all simulations" begin 

for sensealg in sensealgs
    for adtype in adtypes
        
        print("\nTesting\n")
        @show sensealg
        @show adtype
        print("\n")

        # Define dummy grad
        # dummy_grad = function (du, u; simulation::Union{FunctionalInversion, Nothing}=nothing)
        #     du .= maximum(abs.(u)) .* rand(Float64, size(u))
        # end

        params = Parameters(simulation = SimulationParameters(working_dir=working_dir,
                                                            use_MB=false,
                                                            velocities=true,
                                                            tspan=(2010.0, 2015.0),
                                                            multiprocessing=false,
                                                            workers=1,
                                                            test_mode=true,
                                                            rgi_paths=rgi_paths),
                            hyper = Hyperparameters(batch_size=4,
                                                    epochs=10,
                                                    optimizer=ODINN.ADAM(0.01)),
                            UDE = UDEparameters(sensealg=sensealg, 
                                                optim_autoAD=adtype, 
                                                # grad=dummy_grad, 
                                                grad=nothing, 
                                                optimization_method="AD+AD",
                                                target = "A")
                            )

        ## Retrieving simulation data for the following glaciers
        rgi_ids = ["RGI60-11.03638", "RGI60-11.01450"]#, "RGI60-08.00213", "RGI60-04.04351"]

        model = Model(iceflow = SIA2Dmodel(params),
                        mass_balance = mass_balance = TImodel1(params; DDF=6.0/1000.0, acc_factor=1.2/1000.0),
                        machine_learning = NeuralNetwork(params))

        # We retrieve some glaciers for the simulation
        glaciers = initialize_glaciers(rgi_ids, params)

        # We create an ODINN prediction
        functional_inversion = FunctionalInversion(model, glaciers, params)

        # We run the simulation
        @testset "Sensitivity analysis" begin
        run!(functional_inversion)
        @test true
        end
        
    end
end

end