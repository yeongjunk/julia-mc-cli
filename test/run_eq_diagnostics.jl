using CSV
using TOML
using JuliaMCCLI

include("./model.jl")
include("./io.jl")

import .CLIIO

const FN_DIAG_REPLICAS = "diagnostics_replicas.csv"
const FN_DIAG_WALKERS = "diagnostics_walkers.csv"
const FN_DIAG_EXCHANGE = "diagnostics_exchange.csv"
const FN_DIAG_OVERVIEW = "diagnostics_overview.toml"
const USAGE_STRING = "Usage: julia cli/run_eq_diagnostics.jl --infile <raw.jld2> --outdir <output_dir>"

function parse_args(args)
    length(args) == 4 || error(USAGE_STRING)
    args[1] == "--infile" || error(USAGE_STRING)
    args[3] == "--outdir" || error(USAGE_STRING)
    return args[2], args[4]
end

function save_eq_diagnostics(outdir::AbstractString, diagnostics::JuliaMCCLI.DiagnosticsOverview)
    mkpath(outdir)

    replicas_file = joinpath(outdir, FN_DIAG_REPLICAS)
    walkers_file = joinpath(outdir, FN_DIAG_WALKERS)
    exchange_file = joinpath(outdir, FN_DIAG_EXCHANGE)
    overview_file = joinpath(outdir, FN_DIAG_OVERVIEW)

    CSV.write(replicas_file, JuliaMCCLI.summarize_replica_statistics(diagnostics.replica_diagnostics))
    CSV.write(walkers_file, JuliaMCCLI.summarize_walker_statistics(diagnostics.walker_diagnostics))
    CSV.write(exchange_file, JuliaMCCLI.summarize_exchange_statistics(diagnostics.equilibration_diagnostics))

    open(overview_file, "w") do io
        TOML.print(io, JuliaMCCLI.summarize_overview(diagnostics))
    end

    return replicas_file, walkers_file, exchange_file, overview_file
end

function main(args=ARGS)
    infile, outdir = parse_args(args)

    @info "Loading raw equilibration data" infile
    eq_data = CLIIO.load_equilibration(infile)
    diagnostics = JuliaMCCLI.run_eq_diagnostics(eq_data)

    files = save_eq_diagnostics(outdir, diagnostics)
    @info "Diagnostics saved" files
    return files
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
