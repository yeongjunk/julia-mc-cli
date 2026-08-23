module JuliaMCCLI

using LinearAlgebra
using Random
using DataFrames
using Statistics
import LMC
import LMC.LMCPT
import ReplicaExchange as RE

include("configurations.jl")
include("pipeline_data.jl")
include("model.jl")
include("edge_group_nn_nnn.jl")
include("replica_snapshot.jl")
include("run_eq.jl")
include("sample.jl")
include("diagnostics.jl")

using .Configurations: AdaptConfig, ReplicaConfig, EquilibrationConfig, RunEquilibrationConfig, SamplingConfig, RunSamplingConfig
using .PipelineData: EquilibrationData, SamplingData
using .ReplicaSnapshots: ReplicaSnapshot

export Configurations, PipelineData, ReplicaSnapshots
export AdaptConfig, ReplicaConfig, EquilibrationConfig, RunEquilibrationConfig, SamplingConfig, RunSamplingConfig
export EquilibrationData, SamplingData, ReplicaSnapshot
export AbstractModelConfig, model_dim, model_N0, make_problem
export run_eq, run_sample, run_eq_diagnostics
export ReplicaDiagnostics, WalkerDiagnostics, EquilibrationDiagnostics, DiagnosticsOverview
export summarize_replica_statistics, summarize_walker_statistics, summarize_exchange_statistics, summarize_overview

end
