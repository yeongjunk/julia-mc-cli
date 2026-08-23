module Configurations

export SamplingConfig, EquilibrationConfig

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

end
