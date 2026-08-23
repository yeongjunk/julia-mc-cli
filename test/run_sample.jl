using JSON
using JuliaMCCLI

include("./model.jl")
include("./io.jl")

const CFG = JuliaMCCLI.Configurations
import .CLIIO

const USAGE_STRING = "Usage: julia cli/run_sample.jl <config.json> --infile <input.jld2> --outfile <output.jld2>"

function parse_args(args)
    length(args) == 5 || error(USAGE_STRING)
    args[2] == "--infile" || error(USAGE_STRING)
    args[4] == "--outfile" || error(USAGE_STRING)
    return args[1], args[3], args[5]
end

function parse_model_config(dict)
    name = String(get(dict, "name", "generalized_checkerboard"))
    name == "generalized_checkerboard" || error("Unknown model name: $name")
    return CheckerboardConfig(name, Int(get(dict, "L", 5)), Float64(get(dict, "A", 1.0)), Float64(get(dict, "g", 0.01)), Float64(get(dict, "N0", 1.0)), Float64(get(dict, "kappa", 0.01)))
end

function parse_replica_config(dict)
    model = parse_model_config(dict["model"])
    return CFG.ReplicaConfig(model, Float64.(dict["betas"]), Float64(dict["initial_epsilon"]))
end

function parse_sample_eltype(name::AbstractString)
    name == "ComplexF32" && return ComplexF32
    name == "ComplexF64" && return ComplexF64
    error("Unknown sample_eltype: $name")
end

function parse_sampling_config(dict)
    return CFG.SamplingConfig(String.(dict["schedule"]), Int(dict["swap_every"]), Int(dict["partition_every"]), Int(dict["n_sweeps"]), Int(dict["sample_every"]), Int.(dict["beta_indices"]), parse_sample_eltype(String(dict["sample_eltype"])))
end

function parse_run_sampling_config(dict)
    return CFG.RunSamplingConfig(Int(dict["seed"]), parse_replica_config(dict["replica"]), parse_sampling_config(dict["sample"]))
end

function main(args=ARGS)
    config_file, infile, outfile = parse_args(args)
    cfg = parse_run_sampling_config(JSON.parsefile(config_file))

    @info "Loading equilibration data" infile
    eq_data = CLIIO.load_equilibration(infile)
    data = JuliaMCCLI.run_sample(cfg, eq_data)

    mkpath(dirname(outfile))
    CLIIO.save_sampling(outfile, data)
    @info "Saved raw sampling result" outfile
    println(outfile)

    return outfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
