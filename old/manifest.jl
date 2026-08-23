using JSON
using JLD2
using Glob
using ReplicaExchange
using LMC
using Printf


function check_common_cfg(cfg1::NamedTuple, cfg2::NamedTuple)
    if isempty(cfg1)
        true
    else
        (keys(cfg1) == keys(cfg2)) &&
        all(k -> getfield(cfg1, k) == getfield(cfg2, k), keys(cfg2))
    end
end


named_tuple_equality(cfg1, cfg2) =
    all(k -> getfield(cfg1, k) == getfield(cfg2, k), keys(cfg2))


function manifest_file_tag(fn::AbstractString)
    stem = splitext(basename(fn))[1]
    parts = split(stem, "_")

    length(parts) >= 2 ||
        error("Cannot remove the final suffix from filename: $fn")

    return join(parts[1:end-1], "_")
end


function create_manifest_equilibration(outdir)
    fns = sort(glob("*.jld2", outdir))
    outdct = Dict()

    common_cfg = NamedTuple()

    for fn_i in fns
        dct_i = JLD2.load(fn_i)

        cfg_i = dct_i["run_params"]
        cfg_model_i = cfg_i.model

        cfg_i = Base.structdiff(
            cfg_i,
            (;
                model = cfg_i.model,
                tag = cfg_i.tag,
                timestamp = cfg_i.timestamp,
                config_file = cfg_i.config_file,
            ),
        )

        if isempty(common_cfg)
            common_cfg = cfg_i
        elseif Dict(pairs(common_cfg)) != Dict(pairs(cfg_i))
            error("Common equilibration configuration does not match: $(fn_i)")
        end

        outdct[basename(fn_i)] = Dict(pairs(cfg_model_i))
    end

    outdct["common_cfg"] = Dict(pairs(common_cfg))

    return outdct
end


function create_manifest_sample(outdir)
    fns = sort(glob("*.jld2", outdir))
    outdct = Dict()

    common_cfg = NamedTuple()

    for fn_i in fns
        dct_i = JLD2.load(fn_i)

        equilibration_run_params = dct_i["equilibration_run_params"]
        cfg_model_i = equilibration_run_params.model

        sampling_params_i = dct_i["sampling_params"]

        if isempty(common_cfg)
            common_cfg = sampling_params_i
        elseif Dict(pairs(common_cfg)) != Dict(pairs(sampling_params_i))
            error("Common sampling configuration does not match: $(fn_i)")
        end

        outdct[basename(fn_i)] = Dict(pairs(cfg_model_i))
    end

    outdct["common_cfg"] = Dict(pairs(common_cfg))

    return outdct
end


function check_completeness_verbose(dict_man_eq, dict_man_sample)
    eq_fns = [String(k) for k in keys(dict_man_eq) if endswith(String(k), ".jld2")]

    sample_fns = [String(k) for k in keys(dict_man_sample) if endswith(String(k), ".jld2")]

    eq_tags = manifest_file_tag.(eq_fns)
    sample_tags = manifest_file_tag.(sample_fns)

    eq_set = Set(eq_tags)
    sample_set = Set(sample_tags)

    duplicate_eq = [
        tag
        for tag in unique(eq_tags)
        if count(==(tag), eq_tags) > 1
    ]

    duplicate_sample = [
        tag
        for tag in unique(sample_tags)
        if count(==(tag), sample_tags) > 1
    ]

    missing_samples = sort!(collect(setdiff(eq_set, sample_set)))
    missing_equilibrations = sort!(collect(setdiff(sample_set, eq_set)))

    complete =
        isempty(duplicate_eq) &&
        isempty(duplicate_sample) &&
        isempty(missing_samples) &&
        isempty(missing_equilibrations)

    return Dict(
        "complete" => complete,
        "missing_samples" => missing_samples,
        "missing_equilibrations" => missing_equilibrations,
        "duplicate_equilibrations" => duplicate_eq,
        "duplicate_samples" => duplicate_sample,
    )
end


function main(args = ARGS)
    length(args) == 3 || error(
        "Usage: julia create_manifest.jl <equilibration_raw_dir> <sampling_raw_dir> <manifest.json>"
    )

    eq_raw_dir = args[1]
    sample_raw_dir = args[2]
    outfn = args[3]

    dict_man_eq = create_manifest_equilibration(eq_raw_dir)
    dict_man_sample = create_manifest_sample(sample_raw_dir)

    dict_diagnostic = check_completeness_verbose(
        dict_man_eq,
        dict_man_sample,
    )

    open(outfn, "w") do f
        println(f, "{")

        print(f, "    \"equilibration\": ")
        JSON.print(f, dict_man_eq, 4)
        println(f, ",")
        println(f)

        print(f, "    \"sampling\": ")
        JSON.print(f, dict_man_sample, 4)
        println(f, ",")
        println(f)

        print(f, "    \"diagnostic\": ")
        JSON.print(f, dict_diagnostic, 4)
        println(f)

        println(f, "}")
    end

    return outfn
end


main()

