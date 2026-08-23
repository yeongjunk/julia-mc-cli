using LinearAlgebra
using LMC
using LMC.LMCPT
using ReplicaExchange
using Random
using JLD2, Dates
using JSON

BLAS.set_num_threads(1)

include("./model_part.jl")

const LMC_EPSILON_FIELD = Symbol("\u03f5")
const LMC_SIGMA_FIELD = Symbol("\u03c3")
const LMC_PSI_FIELD = Symbol("\u03c8")


######## Config ########

function read_sampling_config(fn)
    raw = JSON.parsefile(fn)

    model_cfg = read_model_config(raw["model"])
    sampling_raw = get(raw, "sampling", Dict())

    n_samples = Int(get(sampling_raw, "n_samples", 20000))
    n_between_samples = Int(get(sampling_raw, "n_between_samples", 1))
    beta_indices = haskey(sampling_raw, "beta_indices") ? Int.(sampling_raw["beta_indices"]) : nothing

    n_samples > 0 || error("sampling.n_samples must be positive.")
    n_between_samples > 0 || error("sampling.n_between_samples must be positive.")

    seed = Int(get(raw, "seed", 1234))
    tag = String(get(raw, "tag", model_cfg.name * "_lmcpt_sampling"))

    return (; model=model_cfg, n_samples, n_between_samples, beta_indices, seed, tag)
end

function load_raw_eq(fn)
    return jldopen(fn, "r") do f
        Dict(k => f[k] for k in keys(f))
    end
end


######## Snapshot compatibility ########

function snapshot_field(x, ascii_name::Symbol, unicode_name::Symbol)
    if hasproperty(x, ascii_name)
        return getproperty(x, ascii_name)
    elseif hasproperty(x, unicode_name)
        return getproperty(x, unicode_name)
    else
        error("Could not find field $(ascii_name) or $(unicode_name).")
    end
end

snapshot_betas(replica_snapshot) = snapshot_field(replica_snapshot, :betas, :βs)
snapshot_epsilons(replica_snapshot) = snapshot_field(replica_snapshot, :epsilons, :ϵs)
snapshot_sigmas(replica_snapshot) = snapshot_field(replica_snapshot, :sigmas, :σs)
snapshot_psis(replica_snapshot) = snapshot_field(replica_snapshot, :psis, :ψs)


######## Replica reconstruction ########

function make_replicas_from_snapshot(
    problem_factory::Function,
    replica_snapshot;
    rng=Random.GLOBAL_RNG,
)
    betas = snapshot_betas(replica_snapshot)
    epsilons = snapshot_epsilons(replica_snapshot)
    sigmas = snapshot_sigmas(replica_snapshot)
    psis = snapshot_psis(replica_snapshot)

    K = length(betas)
    length(epsilons) == K || error("epsilons has inconsistent length.")
    length(sigmas) == K || error("sigmas has inconsistent length.")
    length(psis) == K || error("psis has inconsistent length.")

    params = [
        begin
            problem_k = problem_factory()
            LMCParams(problem_k, betas[k], epsilons[k], sigmas[k])
        end
        for k in 1:K
    ]

    states = [LMCState(copy(psis[k])) for k in 1:K]

    return LMCReplicas(params, states, betas; rng=rng)
end

function make_replicas_from_snapshot(problem::ProblemWithEG, replica_snapshot; rng=Random.GLOBAL_RNG)
    error("Pass a problem_factory instead of a ProblemWithEG. deepcopy/copy of closure-backed problems can share captured scratch buffers.")
end

function snapshot_replicas(replicas::LMCReplicas)
    walkerids = ReplicaExchange.getwalkerids(replicas)
    slots = eachindex(walkerids)

    return (
        betas = copy(ReplicaExchange.getbetas(replicas)),
        epsilons = [getfield(replicas.params[slot], LMC_EPSILON_FIELD) for slot in slots],
        sigmas = [getfield(replicas.params[slot], LMC_SIGMA_FIELD) for slot in slots],
        energies = [ReplicaExchange.getenergy(replicas, slot) for slot in slots],
        psis = [copy(ReplicaExchange.getstate(replicas, slot)) for slot in slots],
        walkerids = copy(walkerids),
    )
end


######## Sampling ########

function make_sampling_exchangeparams(exchangeparams::ExchangeParams, n_between_samples::Int)
    # One sampling sweep keeps the equilibration exchange cadence, but replaces
    # the very large equilibration n_sweeps with n_between_samples.
    return ExchangeParams(
        exchangeparams.edge_groups,
        exchangeparams.n_between_exchange,
        n_between_samples,
    )
end

function sample_replicas!(replicas::LMCReplicas, exchangeparams; n_samples::Int, beta_indices::Vector{Int}, rng=Random.GLOBAL_RNG)
    K = length(replicas)
    n_samples > 0 || error("n_samples must be positive.")
    all(k -> 1 <= k <= K, beta_indices) || error("beta_indices out of range.")

    betas = ReplicaExchange.getbetas(replicas)
    sampled_betas = collect(betas[beta_indices])

    first_state = ReplicaExchange.getstate(replicas, beta_indices[1])
    dim = length(first_state)
    n_beta = length(beta_indices)

    samples = Array{ComplexF32}(undef, dim, n_beta, n_samples)
    sampled_energies = Matrix{Float64}(undef, n_beta, n_samples)

    for s in 1:n_samples
        equilibrate!(replicas, exchangeparams; rng=rng)

        @inbounds for (j, slot) in enumerate(beta_indices)
            psi = ReplicaExchange.getstate(replicas, slot)

            samples[:, j, s] .= ComplexF32.(psi)
            sampled_energies[j, s] = ReplicaExchange.getenergy(replicas, slot)
        end
    end

    return (; samples, sampled_betas, sampled_energies, beta_indices=copy(beta_indices))
end


######## Main ########

function main(args=ARGS)
    length(args) == 3 || error(
        "Usage: julia run_sampling_from_eq_raw.jl <config.json> <equilibration_raw.jld2> <rawdata_outdir>"
    )

    config_file = args[1]
    raw_eq_file = args[2]
    rawdata_outdir = args[3]

    cfg_sample = read_sampling_config(config_file)
    mkpath(rawdata_outdir)

    raw_eq = load_raw_eq(raw_eq_file)
    run_params = raw_eq["run_params"]
    equilibration_exchangeparams = raw_eq["exchangeparams"]
    replica_snapshot = raw_eq["replica_snapshot"]

    rng = Xoshiro(cfg_sample.seed)

    problem_factory = get_problem_factory(cfg_sample.model)
    replicas = make_replicas_from_snapshot(problem_factory, replica_snapshot; rng=rng)

    K = length(snapshot_betas(replica_snapshot))
    beta_indices = isnothing(cfg_sample.beta_indices) ?
        collect(1:K) : cfg_sample.beta_indices

    sampling_exchangeparams = make_sampling_exchangeparams(
        equilibration_exchangeparams,
        cfg_sample.n_between_samples,
    )

    sampling_params = (;
        n_samples=cfg_sample.n_samples,
        n_between_samples=cfg_sample.n_between_samples,
        beta_indices,
        sample_eltype=ComplexF32,
    )

    @info "Start sampling" n_samples=cfg_sample.n_samples n_between_samples=cfg_sample.n_between_samples beta_indices=beta_indices sampled_betas=ReplicaExchange.getbetas(replicas)[beta_indices] sample_eltype=ComplexF32

    @time sampling_result = sample_replicas!(
        replicas,
        sampling_exchangeparams;
        n_samples=cfg_sample.n_samples,
        beta_indices=beta_indices,
        rng=rng,
    )

    @info "Sampling finished."

    outfile = joinpath(rawdata_outdir, "$(cfg_sample.tag)_seed$(cfg_sample.seed)_sampling.jld2")

    final_replica_snapshot = snapshot_replicas(replicas)

    jldsave(outfile;
        sampling_config_file = config_file,
        sampling_params = sampling_params,
        equilibration_raw_file = raw_eq_file,
        equilibration_run_params = run_params,
        equilibration_exchangeparams = equilibration_exchangeparams,
        sampling_exchangeparams = sampling_exchangeparams,
        sampling_result = sampling_result,
        final_replica_snapshot = final_replica_snapshot,
    )

    @info "Saved sampling raw data" outfile
    println(outfile)
    return outfile
end

main()
