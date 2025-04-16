function replace_includes(str)

    included = ["forward_simulation.jl", "functional_inversion.jl"] # TODO: add new tutorials as they come

    # Here the path loads the files from their proper directory,
    # which may not be the directory of the `examples.jl` file!
    path = "src/tutorials/"

    for ex in included
        content = read(path*ex, String)
        str = replace(str, "include(\"$(ex)\")" => content)
    end
    return str
end