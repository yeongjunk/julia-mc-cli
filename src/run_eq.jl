import .Configurations as CFG
import .PipelineData as PD
import .ReplicaSnapshots as RS

BLAS.set_num_threads(1)

######## Replica setup ########

function random_initial_states(dim::Int, K::Int, N0; rng=Random.GLOBAL_RNG, Ftype=ComplexF64)
    psi_initial = rand(rng, Ftype, dim, K)

    for psi_i in eachcol(psi_initial)
        normalize!(psi_i)
        psi_i .*= sqrt(N0)
    end

    return psi_initial
end

compute_sigma(epsilon, beta) = sqrt(2 * epsilon / beta)

function get_lmc_adapt_params(cfg::CFG.AdaptConfig)
    return LMCPT.LMCAdaptParams(cfg.stop_every, cfg.max_windows, cfg.rate_lower, cfg.rate_upper, cfg.grow)
end

function burnin_and_adapt!(reps::LMCPT.LMCReplicas, adaptparams::LMCPT.LMCAdaptParams; n_round::Int=3, n_burnin::Int=3000, rng=Random.GLOBAL_RNG)
    rngs = [Xoshiro(rand(rng, UInt)) for _ in 1:length(reps)]

    history = Vector{NamedTuple}(undef, n_round)
    for round in 1:n_round
        LMCPT.burnin!(reps, n_burnin, rngs=rngs)
        history[round] = LMCPT.adapt_replicas!(reps, adaptparams, rngs=rngs)
    end
    is_success = all(history[end].success)

    return is_success, history
end

function burnin_and_adapt_replicas!(reps::LMCPT.LMCReplicas, cfg::CFG.AdaptConfig; rng::AbstractRNG=Xoshiro())
    adaptparams = get_lmc_adapt_params(cfg)
    adapt_success, adapt_history = burnin_and_adapt!(reps, adaptparams; n_burnin=cfg.n_burnin, n_round=cfg.n_round, rng=rng)

    adapt_success || error("Adaptation failed")
    return adapt_success, adapt_history
end

function make_replicas(problem_factory::FP, psi_init::AbstractMatrix, betas::Vector{F}, epsilons::AbstractVector{F}; rng=Random.GLOBAL_RNG) where {F,FP}
    K = length(betas)
    size(psi_init, 2) == K || error("psi_init must have one column per beta.")

    params = [
        begin
            beta_k = betas[k]
            epsilon_k = epsilons[k]
            problem_k = problem_factory()
            sigma = compute_sigma(epsilon_k, beta_k)
            LMC.LMCParams(problem_k, beta_k, epsilon_k, sigma)
        end
        for k in 1:K
    ]

    states = [LMC.LMCState(psi_init[:, k]) for k in 1:K]
    return LMCPT.LMCReplicas(params, states, betas, rng=rng)
end

function make_replicas(problem_factory, psi_init, betas, epsilon::F; rng=Random.GLOBAL_RNG) where {F<:Real}
    return make_replicas(problem_factory, psi_init, betas, fill(epsilon, length(betas)); rng=rng)
end

function make_replicas(cfg::CFG.ReplicaConfig; rng=Xoshiro(), Ftype=Float64)
    betas = Ftype.(cfg.betas)
    epsilon = Ftype(cfg.initial_epsilon)
    n_reps = length(betas)

    model_cfg = cfg.model
    factory = problem_factory(model_cfg)
    psi_init = random_initial_states(model_dim(model_cfg), n_reps, model_N0(model_cfg); rng=rng, Ftype=Complex{Ftype})

    return make_replicas(factory, psi_init, betas, epsilon; rng=rng)
end

function make_exchange_params(reps, cfg::CFG.EquilibrationConfig)
    K = length(reps)
    edge_groups, _ = scheduled_edge_groups(K, cfg.schedule)
    return RE.ExchangeParams(edge_groups, cfg.swap_every)
end

function equilibrate_replicas!(reps, cfg::CFG.EquilibrationConfig; rng)
    exchange_params = make_exchange_params(reps, cfg)
    equilibration_params = RE.EquilibrationParams(cfg.n_sweeps, cfg.partition_every)
    return RE.monitor_equilibration!(reps, equilibration_params, exchange_params; rng=rng)
end

######## Run equilibration ########

function run_eq(cfg::CFG.RunEquilibrationConfig)
    rng = Xoshiro(cfg.seed)

    reps = make_replicas(cfg.replica; rng=rng)
    @info "Replica successfully created."

    @time adapt_success, adapt_history = burnin_and_adapt_replicas!(reps, cfg.adapt; rng=rng)
    @info "Adaptation successful. Start equilibration."

    @time result = equilibrate_replicas!(reps, cfg.equilibration; rng=rng)
    @info "Equilibration finished."

    snapshot = RS.ReplicaSnapshot(reps)
    return PD.EquilibrationData(snapshot, result, cfg, adapt_success, adapt_history)
end
