using LinearAlgebra
using Dates
using JLD2

using GPLattices.CheckerboardSOSMap
using GPLattices.SOSOnsiteGP
using SphereWirtingerOptim

BLAS.set_num_threads(1)

Base.@kwdef struct RefineSampleConfig
    do_armijo::Bool = true
    do_lbfgs::Bool = true
    gd_l::Float64 = 1.0
    gd_steps::Int = 50_000
    gd_tol::Float64 = 1e-12
    gd_ladder::Vector{Float64} = [1.0, 1e-2, 1e-4, 1e-6]
    lbfgs_g_tol::Float64 = 1e-12
    lbfgs_iterations::Int = 10_000
    lbfgs_show_trace::Bool = false
end

function load_jld2_dict(fn)
    return jldopen(fn, "r") do f
        Dict(k => f[k] for k in keys(f))
    end
end

function selected_indices(selection, n::Int)
    selection === :all && return collect(1:n)
    return collect(selection)
end

function output_refined_sample_file(infile; tag = "refined")
    dir = dirname(infile)
    base = splitext(basename(infile))[1]
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    return joinpath(dir, "$(base)_$(tag)_$(stamp).jld2")
end

function run_param_kappa(run_params)
    if hasproperty(run_params, :kappa)
        return run_params.kappa
    elseif hasproperty(run_params, :κ)
        return getproperty(run_params, :κ)
    else
        error("Could not find kappa in run_params.")
    end
end

function run_param_lambda(run_params)
    if hasproperty(run_params, :lambda)
        return run_params.lambda
    else
        return 1.0
    end
end

function checkerboard_refine_model(run_params)
    L = run_params.L
    A = run_params.A
    n_sites = 2 * L^2
    n_constraints = L^2

    Lmap = checkerboard_sos_linearmap(L; A = A)
    buffer = zeros(ComplexF64, n_constraints)

    gp = SOSOnsiteGPModel(
        n_sites,
        n_constraints,
        Lmap,
        run_param_lambda(run_params),
        run_params.g,
        run_param_kappa(run_params),
        run_params.N0,
        buffer,
    )

    grad_psi_star! = getproperty(SOSOnsiteGP, Symbol("grad_\u03c8star!"))

    energy(psi) = SOSOnsiteGP.energy(gp, psi)
    grad!(G, psi) = grad_psi_star!(gp, G, psi)

    return (; L, A, gp, energy, grad!)
end

function refine_one_sample(psi0, model, cfg::RefineSampleConfig)
    psi = copy(psi0)
    normalize!(psi)

    E_initial = model.energy(psi)

    refined = SphereWirtingerOptim.refine_on_sphere!(
        psi,
        model.energy,
        model.grad!;
        do_armijo = cfg.do_armijo,
        do_lbfgs = cfg.do_lbfgs,
        gd_l = cfg.gd_l,
        gd_ladder = cfg.gd_ladder,
        gd_steps = cfg.gd_steps,
        gd_tol = cfg.gd_tol,
        lbfgs_g_tol = cfg.lbfgs_g_tol,
        lbfgs_iterations = cfg.lbfgs_iterations,
        lbfgs_show_trace = cfg.lbfgs_show_trace,
    )

    return (;
        psi = refined.psi,
        E_initial,
        E_final = refined.E_final,
        mu = refined.mu,
        armijo_tangent_grad_norm = refined.armijo_tangent_grad_norm,
        lbfgs_tangent_grad_norm = refined.lbfgs_tangent_grad_norm,
        tangent_grad_norm = isfinite(refined.lbfgs_tangent_grad_norm) ?
            refined.lbfgs_tangent_grad_norm :
            refined.armijo_tangent_grad_norm,
        lbfgs_res = refined.lbfgs_res,
    )
end

function refine_samples_file(
    infile;
    outfile = output_refined_sample_file(infile),
    beta_indices = :all,
    sample_indices = :all,
    cfg::RefineSampleConfig = RefineSampleConfig(),
)
    data = load_jld2_dict(infile)
    run_params = data["equilibration_run_params"]
    sampling_result = data["sampling_result"]
    samples = sampling_result.samples

    ndims(samples) == 3 || error("Expected sampling_result.samples to have shape (dim, n_beta, n_samples).")

    dim, n_beta, n_sample = size(samples)
    beta_sel = selected_indices(beta_indices, n_beta)
    sample_sel = selected_indices(sample_indices, n_sample)

    model = checkerboard_refine_model(run_params)

    refined_samples = Array{eltype(samples)}(undef, dim, length(beta_sel), length(sample_sel))
    E_initial = Matrix{Float64}(undef, length(beta_sel), length(sample_sel))
    E_final = similar(E_initial)
    mu = similar(E_initial)
    armijo_tangent_grad_norm = similar(E_initial)
    lbfgs_tangent_grad_norm = similar(E_initial)
    tangent_grad_norm = similar(E_initial)
    lbfgs_res = Matrix{Any}(undef, length(beta_sel), length(sample_sel))

    for (jb, beta_idx) in pairs(beta_sel)
        for (js, sample_idx) in pairs(sample_sel)
            @info "Refining sample" beta_idx sample_idx
            result = refine_one_sample(view(samples, :, beta_idx, sample_idx), model, cfg)

            refined_samples[:, jb, js] .= result.psi
            E_initial[jb, js] = result.E_initial
            E_final[jb, js] = result.E_final
            mu[jb, js] = result.mu
            armijo_tangent_grad_norm[jb, js] = result.armijo_tangent_grad_norm
            lbfgs_tangent_grad_norm[jb, js] = result.lbfgs_tangent_grad_norm
            tangent_grad_norm[jb, js] = result.tangent_grad_norm
            lbfgs_res[jb, js] = result.lbfgs_res
        end
    end

    refined_result = (;
        samples = refined_samples,
        beta_indices = beta_sel,
        sample_indices = sample_sel,
        sampled_betas = sampling_result.sampled_betas[beta_sel, sample_sel],
        source_sampled_energies = sampling_result.sampled_energies[beta_sel, sample_sel],
        E_initial,
        E_final,
        mu,
        armijo_tangent_grad_norm,
        lbfgs_tangent_grad_norm,
        tangent_grad_norm,
        lbfgs_res,
    )

    mkpath(dirname(outfile))
    jldsave(outfile;
        source_file = infile,
        source_basename = basename(infile),
        refined_at = string(now()),
        refine_config = cfg,
        equilibration_run_params = run_params,
        sampling_params = get(data, "sampling_params", nothing),
        refined_result = refined_result,
    )

    @info "Saved refined samples" outfile
    println(outfile)
    return outfile
end

function main(args = ARGS)
    length(args) >= 1 || error("Usage: julia refine_sample.jl <sample_raw.jld2> [outfile.jld2]")

    infile = args[1]
    outfile = length(args) >= 2 ? args[2] : output_refined_sample_file(infile)

    return refine_samples_file(infile; outfile)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
