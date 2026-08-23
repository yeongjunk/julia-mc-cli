using JLD2, Dates
# using CairoMakie
# using LaTeXStrings
using DataFrames, CSV
using Statistics
using LinearAlgebra
using TOML


# Needed for JLD2 deserialization of saved structs.
import LMC as LMC
import LMC.LMCPT as LMCPT
import ReplicaExchange as RE

include("replica_snapshot.jl")
include("io.jl")

import .ReplicaSnapshots as RS
import .EquilibrationIO as EqIO 

struct ReplicaDiagnostics{S}
    energies::Matrix{Float64}       # `energies[block, replica]`; see `monitor_equilibration!`
    snapshot::RS.ReplicaSnapshot{S}         # Final replica states and parameters; see `ReplicaSnapshots`
    final_acceptance_rates::RE.AcceptanceRates # `rate[replica]`; see `RE.compute_acceptance_rate`
end

struct WalkerDiagnostics
    walker::RE.WalkerStatus # Walker trajectories across temperature slots; see `RE.WalkerStatus`
end

struct EquilibrationDiagnostics{F}
    betas::Vector{F}                                  # Inverse temperatures indexed by slot
    edge_groups::RE.EdgeGroupsOf{RE.Edge}             # Exchange pairs grouped by stage; see exchange scheduler
    exchange_rates::Vector{RE.ExchangeRates}          # `[block][group][edge]`; see `RE.compute_exchange_rate`
    acceptance_rates::Vector{RE.AcceptanceRates}      # `[block][replica]`; see `RE.compute_acceptance_rate`
end

struct DiagnosticsOverview{S,F,A}
    replica_diagnostics::ReplicaDiagnostics{S}          # Replica quantities; see `ReplicaDiagnostics`
    walker_diagnostics::WalkerDiagnostics               # Walker trajectories; see `WalkerDiagnostics`
    equilibration_diagnostics::EquilibrationDiagnostics{F} # Block-resolved rates; see `EquilibrationDiagnostics`
    adapt_success::Bool                                  # Whether step-size adaptation succeeded
    adapt_history::A                                     # Adaptation history; see the adaptation routine
end

function create_diagnostics(eq_data)
    result = eq_data.result
    snapshot = eq_data.snapshot

    equilibration = EquilibrationDiagnostics(result, snapshot)
    replica = ReplicaDiagnostics(result.energies, snapshot, equilibration.acceptance_rates[end])
    walker = WalkerDiagnostics(result.walker[end])

    return DiagnosticsOverview(replica, walker, equilibration, eq_data.adapt_success, eq_data.adapt_history)
end


######## ReplicaExchange diagnostics constructors ########

function EquilibrationDiagnostics(result, snapshot::RS.ReplicaSnapshot)
    exchange_history = result.exchange
    acceptance_history = result.acceptance

    edge_groups = exchange_history[1].edge_groups
    exchange_rates = RE.compute_exchange_rates.(exchange_history)
    acceptance_rates = RE.compute_acceptance_rates.(acceptance_history)

    return EquilibrationDiagnostics(
        snapshot.betas,
        edge_groups,
        exchange_rates,
        acceptance_rates,
    )
end


######## DataFrame summaries ########

function summarize_replica_statistics(diagnostics::ReplicaDiagnostics)
    snapshot = diagnostics.snapshot

    return DataFrame(
        slot_index      = eachindex(snapshot.betas),
        beta            = snapshot.betas,
        walkerid        = snapshot.walkerids,
        epsilon         = snapshot.epsilons,
        sigma           = snapshot.sigmas,
        norm            = norm.(snapshot.states) .^ 2,
        acceptance_rate = diagnostics.final_acceptance_rates,
    )
end


function summarize_walker_statistics(diagnostics::WalkerDiagnostics)
    walker = diagnostics.walker

    return DataFrame(
        walkerid           = eachindex(walker.hot_visits),
        hot_visits         = walker.hot_visits,
        cold_visits        = walker.cold_visits,
        endpoint_crossings = walker.endpoint_crossings,
    )
end


function summarize_exchange_statistics(diagnostics::EquilibrationDiagnostics)
    rows = NamedTuple[]
    betas = diagnostics.betas
    edge_groups = diagnostics.edge_groups

    for block in eachindex(diagnostics.exchange_rates)
        rates = diagnostics.exchange_rates[block]

        for group_index in eachindex(edge_groups)
            edges = edge_groups[group_index]
            group_rates = rates[group_index]

            for edge_index in eachindex(edges)
                slot_i, slot_j = edges[edge_index]

                push!(rows, (
                    block       = block,
                    group_index = group_index,
                    edge_index  = edge_index,
                    slot_i      = slot_i,
                    slot_j      = slot_j,
                    beta_i      = betas[slot_i],
                    beta_j      = betas[slot_j],
                    rate        = group_rates[edge_index],
                ))
            end
        end
    end

    df = DataFrame(rows)
    sort!(df, :rate)
    return df
end

function summarize_overview_exchange(diagnostics::EquilibrationDiagnostics)
    final_rates = diagnostics.exchange_rates[end]
    edge_groups = diagnostics.edge_groups
    n_groups = length(edge_groups)

    rate_min  = Vector{Float64}(undef, n_groups)
    rate_max  = Vector{Float64}(undef, n_groups)
    rate_mean = Vector{Float64}(undef, n_groups)

    rate_minedge = Vector{Tuple{Int, Int}}(undef, n_groups)
    rate_maxedge = Vector{Tuple{Int, Int}}(undef, n_groups)

    for i in 1:n_groups
        rates = final_rates[i]
        edges = edge_groups[i]

        # findmin, findmax는 (최소/최대값, 해당 인덱스) 튜플을 반환합니다.
        min_val, min_idx = findmin(rates)
        max_val, max_idx = findmax(rates)

        rate_min[i] = min_val
        rate_max[i] = max_val
        rate_mean[i] = mean(rates)

        rate_minedge[i] = edges[min_idx]
        rate_maxedge[i] = edges[max_idx]
    end
    
    return (rate_min = rate_min,
            rate_max = rate_max,
            rate_mean = rate_mean,
            rate_minedge = rate_minedge,
            rate_maxedge = rate_maxedge
           )
end

function summarize_overview_exchange_dict(diagnostics::EquilibrationDiagnostics)
    data = summarize_overview_exchange(diagnostics)
    
    dict = Dict(
        "group_$i" => Dict(
            "rate_min"     => data.rate_min[i],
            "rate_max"     => data.rate_max[i],
            "rate_mean"    => data.rate_mean[i],
            "rate_minedge" => collect(data.rate_minedge[i]),
            "rate_maxedge" => collect(data.rate_maxedge[i])
        ) for i in eachindex(diagnostics.edge_groups)
    )
    return sort(dict)
end

function get_acceptance_info(replica_diagnostics, betas)
    min, argmin = findmin(replica_diagnostics.final_acceptance_rates)
    max, argmax = findmax(replica_diagnostics.final_acceptance_rates)
    meanval = mean(replica_diagnostics.final_acceptance_rates)

    return (min=min, 
            argmin=argmin, 
            argmin_beta = betas[argmin], 
            max=max, 
            argmax=argmax, 
            argmax_beta = betas[argmax], 
            meanval=meanval
           )
end

function get_beta_info(equilibration_diagnostics)
    len = length(equilibration_diagnostics.betas)
    min = minimum(equilibration_diagnostics.betas)
    max = maximum(equilibration_diagnostics.betas)

    return (len=len, min=min, max=max)
end

function get_endpoint_crossing_rate(walker_diagnostics)
    walker = walker_diagnostics.walker 
    return count(!iszero, walker.endpoint_crossings)/length(walker.endpoint_crossings)
end


function summarize_overview(diagnostics::DiagnosticsOverview)
    # beta information
    betas =  diagnostics.equilibration_diagnostics.betas
    beta_info = get_beta_info(diagnostics.equilibration_diagnostics)
    acceptance_info = get_acceptance_info(diagnostics.replica_diagnostics, betas)

    # final acceptance rate
    adapt_sucess = diagnostics.adapt_success
    
    endpoint_crossing_rate = get_endpoint_crossing_rate(diagnostics.walker_diagnostics)
    dict_exchange = summarize_overview_exchange_dict(diagnostics.equilibration_diagnostics)
    
    dict_walker = Dict(endpoint_crossing_rate => endpoint_crossing_rate)

    dct_beta = Dict(pairs(beta_info))
    dict_acceptance = Dict(pairs(acceptance_info))
     
    dict_full = Dict("betas" => dct_beta,
        "acceptance_rate_summary" => dict_acceptance,
        "exchange_rate_summary" => dict_exchange)

    return dict_full
end


######## Loading / paths ########


function theoretical_Egs(cfg)
    th = cfg.model.th
    N0 = cfg.model.N0
    g = cfg.model.g
    L = cfg.model.L
    
    n = N0/L
    if th < 1/8
        E_gs = N0*g*n/2*(cos(2*th)^4 + sin(2*th)^4)
    else
        E_gs = N0*g*n/2*0.5
    end
    return E_gs 
end

function flatten_by_block(xs)
    n_blocks = length(xs)
    n_slots = length(first(xs))

    out = Matrix{eltype(first(xs))}(undef, n_blocks, n_slots)

    for i in 1:n_blocks
        out[i, :] .= xs[i]
    end

    return out
end

######## Main processing ########

const FN_DIAG_REPLICAS = "diagnostics_replicas.csv"
const FN_DIAG_WALKERS = "diagnostics_walkers.csv"
const FN_DIAG_EXCHANGE = "diagnostics_exchange.csv"
const FN_DIAG_OVERVIEW = "diagonstics_overview.toml"

function process_file(infile, outdir)
    mkpath(outdir)
    @info "Loading raw equilibration data" infile
    eq_data = EqIO.load_equilibration(infile)
    diagnostics = create_diagnostics(eq_data)

    replicas_file = joinpath(outdir, FN_DIAG_REPLICAS)
    walkers_file = joinpath(outdir, FN_DIAG_WALKERS)
    exchange_file = joinpath(outdir, FN_DIAG_EXCHANGE)
    overview_file = joinpath(outdir, FN_DIAG_OVERVIEW)

    @info "Saving replicas diagnostics" replicas_file
    df_replica = summarize_replica_statistics(diagnostics.replica_diagnostics)
    CSV.write(replicas_file, df_replica)

    @info "Saving walker diagnostics" walkers_file
    df_walker = summarize_walker_statistics(diagnostics.walker_diagnostics)
    CSV.write(walkers_file, df_walker)

    @info "Saving exchange diagnostics" exchange_file
    df_exchange = summarize_exchange_statistics(diagnostics.equilibration_diagnostics)
    CSV.write(exchange_file, df_exchange)

    @info "Saving diagnostics overview toml" 
    # TODO: Generate diagnostic figures.
    dict_overview = summarize_overview(diagnostics)
    open(overview_file, "w") do io
           TOML.print(io, dict_overview)
    end
    
    @info "Diagnostics saved" replicas_file walkers_file exchange_file overview_file

    return replicas_file, walkers_file, exchange_file, overview_file
end

function main(args=ARGS)
    length(args) == 4 || error("Usage: julia eq_diagnostics.jl --infile <raw.jld2> --outfile <output.arrow>")

    infile = args[2]
    outdir = args[4]

    return process_file(infile, outdir)
end

main()
