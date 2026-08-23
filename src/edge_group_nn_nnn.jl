function normalize_exchange_group_name(name::AbstractString)
    return lowercase(replace(strip(name), r"\s+" => "_", "-" => "_"))
end

function exchange_edge_group_pool(k::Int)
    return Dict(
        "nn_odd"  => [(i, i + 1) for i in 1:2:k-1],
        "nn_even" => [(i, i + 1) for i in 2:2:k-1],
        "nnn_0"   => [(i, i + 2) for i in 1:3:k-2],
        "nnn_1"   => [(i, i + 2) for i in 2:3:k-2],
        "nnn_2"   => [(i, i + 2) for i in 3:3:k-2],
    )
end

function scheduled_edge_groups(K::Int, schedule)
    pool = exchange_edge_group_pool(K)
    names = normalize_exchange_group_name.(schedule)
    available_names = collect(keys(pool))
    unknown_names = setdiff(names, available_names)

    if !isempty(unknown_names)
        available = join(sort(available_names), ", ")
        unknown = join(unknown_names, ", ")
        error("Unknown exchange schedule group(s): $unknown. " * "Available groups: $available")
    end

    groups = [pool[name] for name in names]
    return (groups=groups, names=names)
end
