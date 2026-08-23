using GPLattices.SOSOnsiteGP
using GPLattices.CheckerboardSOSMap
using LMC
using JuliaMCCLI: AbstractModelConfig
import JuliaMCCLI: model_dim, model_N0, make_problem

struct CheckerboardConfig <: AbstractModelConfig
    name::String
    L::Int
    A::Float64
    g::Float64
    N0::Float64
    kappa::Float64
end

model_dim(model_cfg::CheckerboardConfig) = 2 * model_cfg.L^2
model_N0(model_cfg::CheckerboardConfig) = model_cfg.N0

function make_problem(model_cfg::CheckerboardConfig)
    L = model_cfg.L
    dim_site = 2 * L^2
    dim_unitcell = L^2
    t = 1.0

    Lmap = checkerboard_sos_linearmap(L; A=model_cfg.A)
    buffer = zeros(ComplexF64, dim_unitcell)
    model_sos = SOSOnsiteGPModel(dim_site, dim_unitcell, Lmap, t, model_cfg.g, model_cfg.kappa, model_cfg.N0, buffer)

    H_cb(psi) = SOSOnsiteGP.energy(model_sos, psi)
    grad_cb!(psi, G) = grad_ψstar!(model_sos, G, psi)
    energy_and_grad_cb!(psi, G) = energy_and_grad_ψstar!(model_sos, G, psi)

    return LMC.ProblemWithEG(grad_cb!, H_cb, energy_and_grad_cb!)
end
