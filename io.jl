module EquilibrationIO

using JLD2

function save_equilibration(
    outfile;
    snapshot,
    result,
    config,
    adapt_success,
    adapt_history,
)
    JLD2.jldsave(
        outfile;
        snapshot,
        result,
        config,
        adapt_success,
        adapt_history,
    )

    return nothing
end

function load_equilibration(infile)
    data = JLD2.load(infile)

    return (
        snapshot=data["snapshot"],
        result=data["result"],
        config=data["config"],
        adapt_success=data["adapt_success"],
        adapt_history=data["adapt_history"],
    )
end


function save_sample(outfile; result, config)
    JLD2.jldsave(outfile; result, config)
    return nothing
end

function load_sample()
end

end
