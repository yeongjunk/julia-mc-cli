using JSON
using JuliaMCCLI

include("./model.jl")
include("./io.jl")

const CFG = JuliaMCCLI.Configurations
import .CLIIO

const USAGE_STRING = "Usage: julia cli/run_eq.jl <config.json> --outfile <output.jld2>"

function parse_model_config(dict)
    name = String(get(dict, "name", "generalized_checkerboard"))
    name == "generalized_checkerboard" || error("Unknown model name: $name")

    return CheckerboardConfig(
        name,
        Int(get(dict, "L", 5)),
        Float64(get(dict, "A", 1.0)),
        Float64(get(dict, "g", 0.01)),
        Float64(get(dict, "N0", 1.0)),
        Float64(get(dict, "kappa", 0.01)),
    )
end

function parse_replica_config(dict)
    model = parse_model_config(dict["model"])
    return CFG.ReplicaConfig(model, Float64.(dict["betas"]), Float64(dict["initial_epsilon"]))
end

function parse_adapt_config(dict)
    return CFG.AdaptConfig(
        Int(dict["stop_every"]),
        Int(dict["max_windows"]),
        Float64(dict["rate_lower"]),
        Float64(dict["rate_upper"]),
        Float64(dict["grow"]),
        Int(dict["n_burnin"]),
        Int(dict["n_round"]),
    )
end

function parse_equilibration_config(dict)
    return CFG.EquilibrationConfig(String.(dict["schedule"]), Int(dict["swap_every"]), Int(dict["partition_every"]), Int(dict["n_sweeps"]))
end

function parse_run_equilibration_config(dict)
    return CFG.RunEquilibrationConfig(
        Int(dict["seed"]),
        parse_replica_config(dict["replica"]),
        parse_adapt_config(dict["adapt"]),
        parse_equilibration_config(dict["equilibration"]),
    )
end

function parse_args(args)
    length(args) == 3 || error(USAGE_STRING)
    args[2] == "--outfile" || error(USAGE_STRING)
    return args[1], args[3]
end

function main(args=ARGS)
    config_file, outfile = parse_args(args)
    cfg = parse_run_equilibration_config(JSON.parsefile(config_file))

    data = JuliaMCCLI.run_eq(cfg)
    mkpath(dirname(outfile))
    CLIIO.save_equilibration(outfile, data)

    @info "Saved raw equilibration result" outfile
    println(outfile)
    return outfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
