using JLD2
using CairoMakie
using LaTeXStrings
using ColorSchemes

include("./plot_base.jl")

function number_of_samples(result)
    lags = result["nematic_autocorrelation"]["lags"]
    n_samples = length(lags)
    n_samples > 0 || error("number of samples must be positive")
    return n_samples
end

standard_error(std_values, n_samples) = std_values ./ sqrt(n_samples)

function plot_density_std_sem(
    output_dict;
    theta_values=collect(0.14:0.01:0.25),
    cmap=:viridis,
)
    results = output_dict["results"]

    θmin = minimum(theta_values)
    θmax = maximum(theta_values)

    fig = Figure(size=(600, 360), fontsize=22)

    ax = Axis(
        fig[2, 1];
        xlabel=L"T=1/\beta",
        ylabel=L"\mathrm{std}(|\psi_l|^2)",
        xscale=log10,
        ygridvisible=false,
        xgridvisible=false,
    )

    color_scheme = ColorSchemes.colorschemes[cmap]

    for theta in theta_values
        result = find_result_by_theta(results, theta)

        betas = Float64.(result["sampled_betas"])
        density_statistics = result["density_statistics"]
        density_std_mean = Float64.(density_statistics["spatial_std_mean"])
        density_std_std = Float64.(density_statistics["spatial_std_std"])

        n_samples = number_of_samples(result)
        density_std_sem = standard_error(density_std_std, n_samples)

        temperatures = 1.0 ./ betas
        perm = sortperm(temperatures)

        temperatures_sorted = temperatures[perm]
        density_std_mean_sorted = density_std_mean[perm]
        density_std_sem_sorted = density_std_sem[perm]

        color_coordinate = (theta - θmin) / (θmax - θmin)
        curve_color = get(color_scheme, color_coordinate)

        band!(
            ax,
            temperatures_sorted,
            max.(density_std_mean_sorted .- density_std_sem_sorted, 0.0),
            density_std_mean_sorted .+ density_std_sem_sorted;
            color=(curve_color, 0.20),
        )

        lines!(
            ax,
            temperatures_sorted,
            density_std_mean_sorted;
            color=curve_color,
            linewidth=2,
        )

        scatter!(
            ax,
            temperatures_sorted,
            density_std_mean_sorted;
            color=curve_color,
            markersize=8,
        )
    end

    Colorbar(
        fig[1, 1];
        limits=(θmin, θmax),
        colormap=cmap,
        vertical=false,
        flipaxis=true,
        label=L"\theta/\pi",
        ticks=[0.14, 0.19, 0.24],
    )

    rowgap!(fig.layout, 8)

    return fig
end

function plot_density_structure_factor_temperature_sem(
    output_dict;
    theta_values=[0.23, 0.25],
    start_idx=10,
    stride=5,
    cmap=:jet,
)
    results = output_dict["results"]

    fig = Figure(size=(600, 600), fontsize=22)
    curve_color = first(Makie.wong_colors())

    for (col, theta) in enumerate(theta_values)
        result = find_result_by_theta(results, theta)

        betas = Float64.(result["sampled_betas"])
        Ts = 1.0 ./ betas

        sf = result["density_structure_factor"]
        q = Float64.(sf["q"])
        sf_mean = Float64.(sf["mean"])
        sf_std = Float64.(sf["std"])

        size(sf_std) == size(sf_mean) ||
            error("density structure-factor std dimension is inconsistent")

        n_samples = number_of_samples(result)
        sf_sem = standard_error(sf_std, n_samples)

        sel = sort(unique(vcat(
            start_idx:stride:length(betas),
            length(betas)-2:length(betas),
        )))
        sel_plot = reverse(sel)

        Tsel = Ts[sel_plot]
        logTsel = log10.(Tsel)
        lo, hi = extrema(logTsel)

        γ = 0.5
        colorfun(T) = ((log10(T) - lo) / (hi - lo))^γ

        pi_idx = argmin(abs.(q .- π))
        Sπ_mean = vec(sf_mean[pi_idx, :])
        Sπ_sem = vec(sf_sem[pi_idx, :])

        ax_top = Axis(
            fig[2, col];
            xlabel=L"q",
            ylabel=L"\langle |n(q)|^2\rangle",
            xgridvisible=false,
            ygridvisible=false,
            title=L"\theta/\pi = %$(theta)",
        )

        vlines!(ax_top, [π]; linestyle=:dash, color=:black)

        for idx in sel_plot
            lines!(
                ax_top,
                q,
                sf_mean[:, idx];
                color=colorfun(Ts[idx]),
                colormap=cmap,
                colorrange=(0, 1),
                linewidth=2,
            )
        end

        ax_top.xticks = (
            [0, π/2, π, 3π/2, 2π],
            ["0", "π/2", "π", "3π/2", "2π"],
        )

        ax_bottom = Axis(
            fig[3, col];
            xlabel=L"T",
            ylabel=L"\langle |n(\pi)|^2\rangle",
            xscale=log10,
            xgridvisible=false,
            ygridvisible=false,
        )

        perm = sortperm(Ts)
        Ts_sorted = Ts[perm]
        Sπ_mean_sorted = Sπ_mean[perm]
        Sπ_sem_sorted = Sπ_sem[perm]

        band!(
            ax_bottom,
            Ts_sorted,
            max.(Sπ_mean_sorted .- Sπ_sem_sorted, 0.0),
            Sπ_mean_sorted .+ Sπ_sem_sorted;
            color=(curve_color, 0.20),
        )

        lines!(
            ax_bottom,
            Ts_sorted,
            Sπ_mean_sorted;
            color=curve_color,
            linewidth=2,
        )

        scatter!(
            ax_bottom,
            Ts_sorted,
            Sπ_mean_sorted;
            color=curve_color,
            markersize=7,
        )
    end

    Colorbar(
        fig[1, 1:2];
        limits=(0, 1),
        colormap=cmap,
        vertical=false,
        label=L"T",
    )

    rowgap!(fig.layout, 10)

    return fig
end

function main(args=ARGS)
    length(args) == 2 || error(
        "Usage: julia run_plot_density_sem.jl " *
        "<analysis_results.jld2> <figure_dir>"
    )

    analysis_file = args[1]
    figure_dir = args[2]

    dict = JLD2.load(analysis_file)
    haskey(dict, "analysis_results") ||
        error("analysis_results key not found in $(analysis_file)")

    output_dict = dict["analysis_results"]
    mkpath(figure_dir)

    density_sf_figure =
        plot_density_structure_factor_temperature_sem(output_dict)
    density_std_figure = plot_density_std_sem(output_dict)

    density_sf_file =
        joinpath(figure_dir, "density_structure_factor_temperature.pdf")
    density_std_file = joinpath(figure_dir, "density_std.pdf")

    save(density_sf_file, density_sf_figure)
    save(density_std_file, density_std_figure)

    @info "Saved density figures with SEM bands" density_sf_file density_std_file

    return nothing
end

main()
