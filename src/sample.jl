import .Configurations as CFG
import .PipelineData as PD
import .ReplicaSnapshots as RS

######## Configuration checks ########

function check_model_equality(eq_cfg::CFG.RunEquilibrationConfig, sample_cfg::CFG.RunSamplingConfig)
    return eq_cfg.replica.model == sample_cfg.replica.model
end

function check_exchange_equality(eq_cfg::CFG.EquilibrationConfig, sample_cfg::CFG.SamplingConfig, K::Int)
    eq_edge_groups, _ = scheduled_edge_groups(K, eq_cfg.schedule)
    sample_edge_groups, _ = scheduled_edge_groups(K, sample_cfg.schedule)
    return eq_cfg.swap_every == sample_cfg.swap_every && eq_edge_groups == sample_edge_groups
end

function validate_sampling_config(eq_data::PD.EquilibrationData, cfg::CFG.RunSamplingConfig)
    eq_cfg = eq_data.config
    all(cfg.replica.betas .== eq_cfg.replica.betas) || error("betas do not match.")
    check_model_equality(eq_cfg, cfg) || error("model configurations do not match.")
    check_exchange_equality(eq_cfg.equilibration, cfg.sampling, length(cfg.replica.betas)) || error("exchange configurations do not match.")
    return nothing
end

######## Sampling ########

function make_exchange_params(reps, cfg::CFG.SamplingConfig)
    K = length(reps)
    edge_groups, _ = scheduled_edge_groups(K, cfg.schedule)
    return RE.ExchangeParams(edge_groups, cfg.swap_every)
end

function sample_replicas!(reps, cfg::CFG.SamplingConfig; rng)
    exchange_params = make_exchange_params(reps, cfg)
    sample_params = RE.SamplingParams(cfg.n_sweeps, cfg.sample_every, cfg.partition_every, cfg.beta_indices)
    return RE.sample_replicas!(reps, sample_params, exchange_params; rng=rng, sample_eltype=cfg.sample_eltype)
end

function run_sample(cfg::CFG.RunSamplingConfig, eq_data::PD.EquilibrationData)
    validate_sampling_config(eq_data, cfg)
    rng = Xoshiro(cfg.seed)

    factory = problem_factory(cfg.replica.model)
    reps = RS.reconstruct_replicas(factory, eq_data.snapshot; rng=rng)
    @info "Replicas reconstructed."

    @time result = sample_replicas!(reps, cfg.sampling; rng=rng)
    @info "Sampling finished."

    return PD.SamplingData(result, cfg)
end
