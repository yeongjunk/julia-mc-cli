ENV["GKSwstype"] = "100"

using Test
using Statistics
using Plots
using JuliaMCCLI

include("gaussian_model.jl")

const PLOT_FILE = joinpath(@__DIR__, "plot.png")

function histogram_density(x; n_bins=35)
    edges = collect(range(minimum(x), maximum(x); length=n_bins + 1))
    counts = zeros(Int, n_bins)

    for value in x
        bin = clamp(searchsortedlast(edges, value), 1, n_bins)
        counts[bin] += 1
    end

    widths = diff(edges)
    centers = @. 0.5 * (edges[1:end-1] + edges[2:end])
    density = counts ./ (length(x) .* widths)
    return centers, density
end

function make_gaussian_configs()
    model = GaussianConfig(1.0, 1.0)
    betas = [0.5, 1.0, 2.0, 4.0]
    replica = ReplicaConfig(model, betas, 0.05)
    schedule = ["nn_even", "nn_odd"]

    adapt = AdaptConfig(500, 30, 0.05, 0.95, 2.0, 500, 2)
    equilibration = EquilibrationConfig(schedule, 5, 2_000, 10_000)
    sampling = SamplingConfig(
        schedule,
        5,
        5_000,
        50_000,
        10,
        [2],
        ComplexF64,
    )

    eq_cfg = RunEquilibrationConfig(1235, replica, adapt, equilibration)
    sample_cfg = RunSamplingConfig(4321, replica, sampling)
    return eq_cfg, sample_cfg
end

@testset "one-variable Gaussian Boltzmann sampling" begin
    eq_cfg, sample_cfg = make_gaussian_configs()
    eq_data = run_eq(eq_cfg)
    sample_data = run_sample(sample_cfg, eq_data)

    samples = sample_data.result.samples
    @test size(samples, 1) == 1
    @test size(samples, 2) == 1

    psi = vec(samples[1, 1, :])
    x = real.(psi)
    y = imag.(psi)

    beta = sample_cfg.replica.betas[2]
    k = sample_cfg.replica.model.k
    exact_variance = 1 / (beta * k)
    exact_mean_energy = 1 / beta
    sampled_energy = @. 0.5 * k * abs2(psi)

    @test abs(mean(x)) < 0.08 * sqrt(exact_variance)
    @test abs(mean(y)) < 0.08 * sqrt(exact_variance)
    @test isapprox(var(x; corrected=false), exact_variance; rtol=0.12)
    @test isapprox(var(y; corrected=false), exact_variance; rtol=0.12)
    @test isapprox(mean(sampled_energy), exact_mean_energy; rtol=0.10)

    centers, density = histogram_density(x)
    x_curve = range(-4sqrt(exact_variance), 4sqrt(exact_variance); length=400)
    exact_density = @. sqrt(beta * k / (2pi)) * exp(-0.5 * beta * k * x_curve^2)

    fig = plot(
        x_curve,
        exact_density;
        label="exact Boltzmann distribution",
        linewidth=2,
        xlabel="Re(psi)",
        ylabel="probability density",
        title="Gaussian test: beta=$(beta), k=$(k)",
        legend=:topright,
    )
    scatter!(
        fig,
        centers,
        density;
        label="LMC samples",
        markersize=4,
        markerstrokewidth=0,
    )
    savefig(fig, PLOT_FILE)

    @test isfile(PLOT_FILE)
    @test filesize(PLOT_FILE) > 0
end
