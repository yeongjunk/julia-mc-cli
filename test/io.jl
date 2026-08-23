module CLIIO

using JLD2
using JuliaMCCLI

const PD = JuliaMCCLI.PipelineData

function save_equilibration(outfile::AbstractString, data::PD.EquilibrationData)
    JLD2.jldsave(outfile; snapshot=data.snapshot, result=data.result, config=data.config, adapt_success=data.adapt_success, adapt_history=data.adapt_history)
    return nothing
end

function load_equilibration(infile::AbstractString)
    data = JLD2.load(infile)
    return PD.EquilibrationData(data["snapshot"], data["result"], data["config"], data["adapt_success"], data["adapt_history"])
end

function save_sampling(outfile::AbstractString, data::PD.SamplingData)
    JLD2.jldsave(outfile; result=data.result, config=data.config)
    return nothing
end

function load_sampling(infile::AbstractString)
    data = JLD2.load(infile)
    return PD.SamplingData(data["result"], data["config"])
end

end
