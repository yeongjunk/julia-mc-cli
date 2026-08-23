using LMC
using JuliaMCCLI: AbstractModelConfig
import JuliaMCCLI: model_dim, model_N0, make_problem

struct GaussianConfig <: AbstractModelConfig
    k::Float64
    initial_norm::Float64
end

model_dim(::GaussianConfig) = 1
model_N0(cfg::GaussianConfig) = cfg.initial_norm

function make_problem(cfg::GaussianConfig)
    k = cfg.k

    energy(psi) = 0.5 * k * sum(abs2, psi)

    function gradient!(psi, G)
        @. G = 0.5 * k * psi
        return nothing
    end

    function energy_and_gradient!(psi, G)
        gradient!(psi, G)
        return energy(psi)
    end

    return LMC.ProblemWithEG(gradient!, energy, energy_and_gradient!)
end
