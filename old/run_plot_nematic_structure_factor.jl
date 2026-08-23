using JLD2
using CairoMakie

include("./plot.jl")
include("./plot_nematic_structure_factor.jl")
include("./plot_phase_diagram.jl")

function main(args=ARGS)
    2 <= length(args) <= 3 || error(
        "Usage: julia run_plot_nematic_structure_factor.jl " *
        "<analysis_results.jld2> <nematic_structure_factor.pdf> " *
        "[density_phase_diagram.pdf]"
    )

    analysis_file = args[1]
    nematic_output = args[2]
    phase_diagram_output = length(args) == 3 ?
        args[3] :
        joinpath(dirname(nematic_output), "density_structure_factor_phase_diagram.pdf")

    dict = JLD2.load(analysis_file)
    haskey(dict, "analysis_results") ||
        error("analysis_results key not found in $(analysis_file)")

    output_dict = dict["analysis_results"]

    nematic_fig = plot_nematic_structure_factor(output_dict)
    phase_diagram_fig = plot_density_structure_factor_pi_phase_diagram(output_dict)

    for output_file in (nematic_output, phase_diagram_output)
        output_dir = dirname(output_file)
        output_dir == "." || mkpath(output_dir)
    end

    save(nematic_output, nematic_fig)
    save(phase_diagram_output, phase_diagram_fig)

    @info "Saved nematic structure-factor figure" output=nematic_output
    @info "Saved density structure-factor phase diagram" output=phase_diagram_output

    println(nematic_output)
    println(phase_diagram_output)

    return (; nematic_fig, phase_diagram_fig)
end

main()
