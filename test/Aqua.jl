function test_Aqua()
    Aqua.test_ambiguities(ODINN)
    Aqua.test_undefined_exports(ODINN)
    Aqua.test_project_extras(ODINN)
    Aqua.test_stale_deps(ODINN; ignore = [
        :JET, :Test, :BenchmarkTools, :Revise, :Aqua, :FiniteDifferences])
    Aqua.test_deps_compat(ODINN)
    Aqua.test_piracies(ODINN; treat_as_own = [:AbstractPrepVJP, :Law, :∂law∂inp!], broken = true)
    Aqua.test_persistent_tasks(ODINN, broken = VERSION < v"1.11") # Precompilation of OrdinaryDiffEq fails for Julia 1.10
    Aqua.test_undocumented_names(ODINN; broken = true)
end
