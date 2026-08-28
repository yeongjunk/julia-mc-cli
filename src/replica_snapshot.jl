module ReplicaSnapshots

using Random: Xoshiro
import LMC
import LMC.LMCPT as LMCPT
import ParallelTemperingSamplers as RE

struct ReplicaSnapshot{TB,Tpsi,TE,TS}
    walkerids::Vector{Int}
    betas::Vector{TB}
    epsilons::Vector{TE}
    sigmas::Vector{TS}
    states::Vector{Vector{Tpsi}}
end

getstates(reps::LMCPT.LMCReplicas) = [RE.getstate(reps, i) for i in 1:length(reps)]
getepsilons(reps::LMCPT.LMCReplicas) = [reps.params[i].ϵ for i in 1:length(reps)]
getsigmas(reps::LMCPT.LMCReplicas) = [reps.params[i].σ for i in 1:length(reps)]

compute_sigma(epsilon, beta) = sqrt(2 * epsilon / beta)

function ReplicaSnapshot(reps::LMCPT.LMCReplicas)
    walkerids = RE.getwalkerids(reps)
    betas = RE.getbetas(reps)
    states = getstates(reps)
    epsilons = getepsilons(reps)
    sigmas = getsigmas(reps)

    return ReplicaSnapshot(walkerids, betas, epsilons, sigmas, states)
end

function reconstruct_replicas(problem_factory::FP, snapshot::ReplicaSnapshot; rng=Xoshiro()) where FP
    states = snapshot.states
    betas = snapshot.betas
    epsilons = snapshot.epsilons
    K = length(betas)

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

    states = [LMC.LMCState(states[k]) for k in 1:K]
    return LMCPT.LMCReplicas(params, states, betas, rng=rng)
end

end
