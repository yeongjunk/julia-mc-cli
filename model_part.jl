using GPLattices.SOSOnsiteGP
using GPLattices.CheckerboardSOSMap
using LMC

model_dim(model_cfg) = 2*model_cfg.L^2
model_N0(model_cfg)  = model_cfg.N0

function read_model_config(model_raw)
    name  = String(get(model_raw, "name", "generalized_checkerboard"))
    L     = Int(get(model_raw, "L", 5))
    A    = Float64(get(model_raw, "A", 1.0))
    g     = Float64(get(model_raw, "g", 0.01))
    N0    = Float64(get(model_raw, "N0", 1.0))
    kappa = Float64(get(model_raw, "kappa", 0.01))
    return (;name=name, L=L, A=A, g=g, N0=N0, kappa=kappa)
end

######## Problem: energy and gradient function #######
function checkerboardgp_problem_generator(;L=5, g=0.5, N0=0.1*2*L^2, A=1.0, kappa=80/*(0.1*2*L^2))
    dim_site = 2*L^2
    dim_unitcell = L^2
    t = 1.0

    Lmap = checkerboard_sos_linearmap(L; A=A)
    buffer = zeros(ComplexF64, dim_unitcell)
    model_sos = SOSOnsiteGPModel(dim_site, dim_unitcell, Lmap, t, g, kappa, N0, buffer)

    H_cb(psi) = SOSOnsiteGP.energy(model_sos, psi)
    grad_cb!(psi, G) = grad_ψstar!(model_sos, G, psi)
    energy_and_grad_cb!(psi, G) = energy_and_grad_ψstar!(model_sos, G, psi)

    return LMC.ProblemWithEG(grad_cb!, H_cb, energy_and_grad_cb!)
end


function get_problem_factory(model_cfg)

    return problem_factory = () -> checkerboardgp_problem_generator(
        L=model_cfg.L,
        N0=model_cfg.N0,
        g=model_cfg.g,
        A=model_cfg.A,
        kappa=model_cfg.kappa,
    )
end
