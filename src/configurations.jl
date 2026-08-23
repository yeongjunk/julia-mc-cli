module Configurations

export AdaptConfig, ReplicaConfig, EquilibrationConfig, RunEquilibrationConfig
export SamplingConfig, RunSamplingConfig

struct AdaptConfig
    stop_every::Int
    max_windows::Int
    rate_lower::Float64
    rate_upper::Float64
    grow::Float64
    n_burnin::Int
    n_round::Int
end

struct ReplicaConfig{M,F<:Real}
    model::M
    betas::Vector{F}
    initial_epsilon::F
end

struct EquilibrationConfig
    schedule::Vector{String}
    swap_every::Int
    partition_every::Int
    n_sweeps::Int
end

struct RunEquilibrationConfig{R<:ReplicaConfig,A<:AdaptConfig,E<:EquilibrationConfig}
    seed::Int
    replica::R
    adapt::A
    equilibration::E
end

struct SamplingConfig
    schedule::Vector{String}
    swap_every::Int
    partition_every::Int
    n_sweeps::Int
    sample_every::Int
    beta_indices::Vector{Int}
    sample_eltype::Type
end

struct RunSamplingConfig{R<:ReplicaConfig,S<:SamplingConfig}
    seed::Int
    replica::R
    sampling::S
end

end
