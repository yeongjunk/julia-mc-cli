using ABF1DGroundstate

model_dim(model_cfg) = model_cfg.L
model_N0(model_cfg)  = model_cfg.N0

function read_model_config(model_raw)
    name  = String(get(model_raw, "name", "abf1d_projected"))
    L     = Int(get(model_raw, "L", 64))
    th    = Float64(get(model_raw, "th", 0.25))
    g     = Float64(get(model_raw, "g", 1.0))
    N0    = Float64(get(model_raw, "N0", 1.0))
    kappa = Float64(get(model_raw, "kappa", 1.0))
    return (;name=name, L=L, th=th, g=g, N0=N0, kappa=kappa)
end

function norm_penalty(psi, N0, kappa)
    N = sum(abs2, psi)
    return kappa * (N - N0)^2
end

function add_norm_penalty_grad!(psi, G, N0, kappa)
    N = sum(abs2, psi)
    c = 2 * kappa * (N - N0)

    @inbounds for i in eachindex(psi, G)
        G[i] += c * psi[i]
    end

    return G
end

######## Problem: energy and gradient function #######
function abf1d_problem_generator(;L=5, N0=L, g=1.0, th=0.25, kappa=5.0)
    n = N0/L # default value n=1
    p = ABF1DParams(L,n,g,th)
    H(psi) = abf1d_projected_H(g, th, psi) + norm_penalty(psi, N0, kappa)

    function grad!(psi, G)
        abf1d_projected_gradient_periodic!(g, th, psi, G)
        add_norm_penalty_grad!(psi, G, N0, kappa)
        return G
    end

    
    function energy_and_grad!(psi, G)  
        grad!(psi, G)
        return H(psi)
    end

    return ProblemWithEG(grad!, H, energy_and_grad!)
end


function get_problem_factory(model_cfg)

    return problem_factory = () -> abf1d_problem_generator(
        L=model_cfg.L,
        N0=model_cfg.N0,
        g=model_cfg.g,
        th=model_cfg.th,
        kappa=model_cfg.kappa,
    )
end
