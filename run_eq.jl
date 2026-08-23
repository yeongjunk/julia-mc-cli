using LinearAlgebra
using Random
using JLD2
using JSON
import LMC
import LMC.LMCPT # LMC and LMC custom package
import ReplicaExchange as RE # ReplicaExchange custom package
using InteractiveUtils
include("./model_part.jl")
include("./edge_group_nn_nnn.jl")
include("./replica_snapshot.jl")
include("./io.jl")

import .ReplicaSnapshots as RS # ./replica_snapshot.jl
import .EquilibrationIO as EqIO # ./io.jl

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

function get_lmc_adapt_params(adapt::AbstractDict)
    return LMCPT.LMCAdaptParams(
        Int(adapt["stop_every"]),
        Int(adapt["max_windows"]),
        Float64(adapt["rate_lower"]),
        Float64(adapt["rate_upper"]),
        Float64(adapt["grow"]),
    )
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

function burnin_and_adapt_replicas!(reps::LMCPT.LMCReplicas, adapt_cfg::AbstractDict; rng::AbstractRNG=Xoshiro())
    # Get parameters
    n_round  = Int(adapt_cfg["n_round"])
    n_burnin = Int(adapt_cfg["n_burnin"])
    adaptparams = get_lmc_adapt_params(adapt_cfg)

    adapt_success, adapt_history = burnin_and_adapt!(reps, adaptparams; n_burnin=n_burnin, n_round=n_round, rng=rng)

    if !adapt_success
        # TODO: save adaptation history metafile
        error("Adaptation failed")
    end

    return adapt_success, adapt_history
end

function make_replicas(problem_factory::FP, psi_init::AbstractMatrix, betas::Vector{F}, epsilons::AbstractVector{F}; rng=Random.GLOBAL_RNG) where {F, FP}
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

function make_replicas(cfg; rng=Xoshiro(), Ftype=Float64)
    betas = Ftype.(cfg["betas"])
    epsilon = Ftype(cfg["initial_epsilon"])
    n_reps = length(betas)

    cfg_model = read_model_config(cfg["model"]) # This part is implemented by the user
    problem_factory = get_problem_factory(cfg_model) # problem_factory() should generate the ProblemWithEG struct in LMC package.
    psi_init = random_initial_states(model_dim(cfg_model), n_reps, model_N0(cfg_model); rng=rng, Ftype=Complex{Ftype}) # model_dim(cfg_model), model_N0(cfg_model) should be implemented by the user

    reps = make_replicas(problem_factory, psi_init, betas, epsilon; rng=rng)

    return reps
end

function make_exchange_params(reps, eq_cfg)
    K = length(reps)
    edge_groups, _ = scheduled_edge_groups(K, eq_cfg["schedule"])

    return RE.ExchangeParams(edge_groups, Int(eq_cfg["swap_every"]))
end

function equilibrate_replicas!(reps, eq_cfg; rng)
    exchange_params = make_exchange_params(reps, eq_cfg)
    equilibration_params = RE.EquilibrationParams(Int(eq_cfg["n_sweeps"]), Int(eq_cfg["partition_every"]))

    return RE.monitor_equilibration!(reps, equilibration_params, exchange_params; rng=rng)
end

function parse_output_file(args)
    length(args) == 3 || error("Usage: julia run_eq.jl <config.json> --outfile <output.jld2>")
    args[2] == "--outfile" || error("Usage: julia run_eq.jl <config.json> --outfile <output.jld2>")

    config_file = args[1]
    outfile = args[3]

    return config_file, outfile
end


######## Main ########

function main(args=ARGS)
    config_file, outfile = parse_output_file(args)

    cfg = JSON.parsefile(config_file)

    replica_cfg = cfg["replica"]
    adapt_cfg = cfg["adapt"]
    eq_cfg = cfg["equilibration"]

    mkpath(dirname(outfile))

    seed = Int(cfg["seed"])
    rng = Xoshiro(seed)

    ######## Create replicas ########
    reps = make_replicas(replica_cfg; rng=rng)
    @info "Replica successuflly created."

    ######## Adaptation ########
    @time adapt_success, adapt_history = burnin_and_adapt_replicas!(reps, adapt_cfg; rng=rng)
    @info "Adaptation successful. Start equilibration."

    ######## Equilibrate ########
    @time result = equilibrate_replicas!(reps, eq_cfg; rng=rng)
    # result.energies, Matrix, (n_reps, n_partitions)
    # result.exchange, Vector{ExchangeStatus}, length(result.exchange) = n_partitions
    # result.acceptance, Vector{AcceptanceStatus}, length(result.acceptance) = n_partitions
    # result.walker, Vector{WalkerStatus}, length(result.walker) = n_partitions
    @info "Equilibration finished."

    ######## Save ########
    snapshot = RS.ReplicaSnapshot(reps)

    EqIO.save_equilibration(outfile;
        snapshot=snapshot,
        result=result,
        config=cfg,
        adapt_success=adapt_success,
        adapt_history=adapt_history,
    )
    @info "Saved raw equilibration result" outfile
    println(outfile)

    return outfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
