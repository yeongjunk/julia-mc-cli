import .PipelineData as PD
import .ReplicaSnapshots as RS

struct ReplicaDiagnostics{S}
    energies::Matrix{Float64}
    snapshot::RS.ReplicaSnapshot{S}
    final_acceptance_rates::RE.AcceptanceRates
end

struct WalkerDiagnostics
    walker::RE.WalkerStatus
end

struct EquilibrationDiagnostics{F}
    betas::Vector{F}
    edge_groups::RE.EdgeGroupsOf{RE.Edge}
    exchange_rates::Vector{RE.ExchangeRates}
    acceptance_rates::Vector{RE.AcceptanceRates}
end

struct DiagnosticsOverview{S,F,A}
    replica_diagnostics::ReplicaDiagnostics{S}
    walker_diagnostics::WalkerDiagnostics
    equilibration_diagnostics::EquilibrationDiagnostics{F}
    adapt_success::Bool
    adapt_history::A
end

######## Diagnostics constructors ########

function EquilibrationDiagnostics(result, snapshot::RS.ReplicaSnapshot)
    exchange_history = result.exchange
    acceptance_history = result.acceptance

    edge_groups = exchange_history[1].edge_groups
    exchange_rates = RE.compute_exchange_rates.(exchange_history)
    acceptance_rates = RE.compute_acceptance_rates.(acceptance_history)

    return EquilibrationDiagnostics(snapshot.betas, edge_groups, exchange_rates, acceptance_rates)
end

function run_eq_diagnostics(eq_data::PD.EquilibrationData)
    result = eq_data.result
    snapshot = eq_data.snapshot

    equilibration = EquilibrationDiagnostics(result, snapshot)
    replica = ReplicaDiagnostics(result.energies, snapshot, equilibration.acceptance_rates[end])
    walker = WalkerDiagnostics(result.walker[end])

    return DiagnosticsOverview(replica, walker, equilibration, eq_data.adapt_success, eq_data.adapt_history)
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

    rate_min = Vector{Float64}(undef, n_groups)
    rate_max = Vector{Float64}(undef, n_groups)
    rate_mean = Vector{Float64}(undef, n_groups)
    rate_minedge = Vector{Tuple{Int,Int}}(undef, n_groups)
    rate_maxedge = Vector{Tuple{Int,Int}}(undef, n_groups)

    for i in 1:n_groups
        rates = final_rates[i]
        edges = edge_groups[i]
        min_val, min_idx = findmin(rates)
        max_val, max_idx = findmax(rates)

        rate_min[i] = min_val
        rate_max[i] = max_val
        rate_mean[i] = mean(rates)
        rate_minedge[i] = edges[min_idx]
        rate_maxedge[i] = edges[max_idx]
    end

    return (rate_min=rate_min, rate_max=rate_max, rate_mean=rate_mean, rate_minedge=rate_minedge, rate_maxedge=rate_maxedge)
end

function summarize_overview_exchange_dict(diagnostics::EquilibrationDiagnostics)
    data = summarize_overview_exchange(diagnostics)

    dict = Dict(
        "group_$i" => Dict(
            "rate_min" => data.rate_min[i],
            "rate_max" => data.rate_max[i],
            "rate_mean" => data.rate_mean[i],
            "rate_minedge" => collect(data.rate_minedge[i]),
            "rate_maxedge" => collect(data.rate_maxedge[i]),
        ) for i in eachindex(diagnostics.edge_groups)
    )

    return dict
end

function get_acceptance_info(replica_diagnostics, betas)
    min, argmin = findmin(replica_diagnostics.final_acceptance_rates)
    max, argmax = findmax(replica_diagnostics.final_acceptance_rates)
    meanval = mean(replica_diagnostics.final_acceptance_rates)

    return (min=min, argmin=argmin, argmin_beta=betas[argmin], max=max, argmax=argmax, argmax_beta=betas[argmax], meanval=meanval)
end

function get_beta_info(equilibration_diagnostics)
    len = length(equilibration_diagnostics.betas)
    min = minimum(equilibration_diagnostics.betas)
    max = maximum(equilibration_diagnostics.betas)

    return (len=len, min=min, max=max)
end

function get_endpoint_crossing_rate(walker_diagnostics)
    walker = walker_diagnostics.walker
    return count(!iszero, walker.endpoint_crossings) / length(walker.endpoint_crossings)
end

function summarize_overview(diagnostics::DiagnosticsOverview)
    betas = diagnostics.equilibration_diagnostics.betas
    beta_info = get_beta_info(diagnostics.equilibration_diagnostics)
    acceptance_info = get_acceptance_info(diagnostics.replica_diagnostics, betas)
    endpoint_crossing_rate = get_endpoint_crossing_rate(diagnostics.walker_diagnostics)
    dict_exchange = summarize_overview_exchange_dict(diagnostics.equilibration_diagnostics)

    dict_walker = Dict("endpoint_crossing_rate" => endpoint_crossing_rate)
    dict_beta = Dict(pairs(beta_info))
    dict_acceptance = Dict(pairs(acceptance_info))

    return Dict(
        "betas" => dict_beta,
        "acceptance_rate_summary" => dict_acceptance,
        "exchange_rate_summary" => dict_exchange,
        "walker_summary" => dict_walker,
        "adapt_success" => diagnostics.adapt_success,
    )
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
