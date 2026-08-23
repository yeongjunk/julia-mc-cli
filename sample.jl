using LinearAlgebra
using Random
using JLD2
using JSON
import LMC
import LMC.LMCPT # LMC and LMC custom package
import ReplicaExchange as RE # ReplicaExchange custom package


include("./model_part.jl")
include("./edge_group_nn_nnn.jl")
include("./replica_snapshot.jl")
include("./io.jl")

import .ReplicaSnapshots as RS # ./replica_snapshot.jl
import .EquilibrationIO as EqIO #./io.jl

BLAS.set_num_threads(1)

struct SamplingConfig
    schedule::Vector{String}
    swap_every::Int
    partition_every::Int
    n_sweeps::Int
    beta_indices::Vector{Int}
    sample_eltype::Type
end

struct ReplicaConfig{T}
    model::T
    betas::Vector{Float64}
end

function get_samplingconfig(dict)
    schedule = dict["schedule"]
    swap_every = dict["swap_every"]
    partition_every = dict["partition_every"]
    n_sweeps = dict["n_sweeps"]
    beta_indices = dict["beta_indices"]
    sample_eltype = dict["sample_eltype"]

    return SamplingConfig(schedule, swap_every, partition_every, n_sweeps, beta_indices, sample_eltype)
end

function get_replicaconfig(dict)
    model = get_model(dict["model"])
    betas = dict["betas"]
    return ReplicaConfig(model, betas)
end


######## configuration equality ########
check_model_equality(model_eq_cfg, model_sample_cfg) = model_sample_cfg == model_eq_cfg

function check_exchange_equality(eq_cfg, sample_cfg, K)
     
    eq_edge_groups, _ = scheduled_edge_groups(K, eq_cfg["schedule"])
    sample_edge_groups, _ = scheduled_edge_groups(K, sample_cfg["schedule"])

    check1 = eq_cfg["swap_every"] == sample_cfg["swap_every"]
    check2 = eq_edge_groups == sample_edge_groups

    return check1 && check2
end


function make_exchange_params(reps, eq_cfg)
    K = length(reps)
    edge_groups, _ = scheduled_edge_groups(K, eq_cfg["schedule"])

    return RE.ExchangeParams(edge_groups, Int(eq_cfg["swap_every"]))
end

function sample_replicas!(reps, sample_cfg; rng)
    exchange_params = make_exchange_params(reps, sample_cfg)

    n_sweeps = Int(sample_cfg["n_sweeps"])
    sample_every = Int(sample_cfg["sample_every"])
    partition_every = Int(sample_cfg["partition_every"])
    beta_indices = Int.(sample_cfg["beta_indices"])

    sample_eltype = eval(Symbol(sample_cfg["sample_eltype"]))

    sample_params =  RE.SamplingParams(n_sweeps, sample_every, partition_every, beta_indices)

    return RE.sample_replicas!(reps, sample_params, exchange_params; rng=rng, sample_eltype=sample_eltype)
end

const USAGE_STRING = "Usage: julia run_eq.jl <config.json> --infile <input.jld2> --outfile <output.jld2>"

function parse_output_file(args)

    length(args) == 5 || error(USAGE_STRING)
    args[2] == "--infile" || error(USAGE_STRING)
    args[4] == "--outfile" || error(USAGE_STRING)

    config_file = args[1]
    infile = args[3]
    outfile = args[5]
    
    return config_file, infile, outfile
end


######## Main ########
function main(args=ARGS)
    config_file, infile, outfile = parse_output_file(args)

    eq_data = EqIO.load_equilibration(infile)
    cfg_of_eq = eq_data.config
    eq_cfg = eq_data.config["equilibration"]

    # Load sampling configurations
    cfg = JSON.parsefile(config_file)

    replica_cfg = get_replicaconfig(cfg["replica"])
    sample_cfg = get_sampleconfig(cfg["sample"])

    betas = replica_cfg.betas
    eq_betas = cfg_of_eq["replica"]["betas"]
    all(betas .== eq_betas) || error("betas do not match.")
    K = length(betas)
    check_exchange_equality(eq_cfg, sample_cfg, K) # compare equilibration and sampling equilibration configuration. 


    mkpath(dirname(outfile))

    seed = Int(cfg["seed"])
    rng = Xoshiro(seed)

    # reconstruct model part
    model_cfg = replica_cfg.model # This part is implemented by the user
    problem_factory = get_problem_factory(model_cfg) # problem_factory() should generate the ProblemWithEG struct in LMC package.

    # recosntruct replicas
    reps = RS.reconstruct_replicas(problem_factory, eq_data.snapshot, rng=rng)  
    @info "Reconstruct replicas."

    
    ######## Save ########
    result = sample_replicas!(reps, sample_cfg, rng=rng)
    @time result = sample_replicas!(reps, sample_cfg, rng=rng)

    EqIO.save_sample(outfile, result=result, config=cfg)
    @info "Saved sampling result" outfile

    return outfile
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
