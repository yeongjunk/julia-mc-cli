function get_arg(args, name, default)
    key = "--" * name

    for i in eachindex(args)
        if args[i] == key && i < length(args)
            return parse(typeof(default), args[i + 1])
        elseif startswith(args[i], key * "=")
            return parse(typeof(default), split(args[i], "=", limit=2)[2])
        end
    end

    return default
end

function interpolate_power(x, y, t, q)
    q == 0 && return exp((1 - t) * log(x) + t * log(y))
    return ((1 - t) * x^q + t * y^q)^(1 / q)
end

function geo_skeleton_power_fill(β_min, β_max; n_geo::Int=9, n_fill::Int=6, q=1.0)
    n_geo >= 1 || error("--n-geo must be at least 1.")
    n_fill >= 1 || error("--n-fill must be at least 1.")
    β_min > 0 || error("--min must be positive.")
    β_max >= β_min || error("--max must be >= --min.")

    βgeo = exp.(range(log(float(β_min)), log(float(β_max)), length=n_geo + 1))
    βs = Float64[]

    for j in 1:n_geo
        x, y = βgeo[j], βgeo[j + 1]

        for k in 0:n_fill-1
            t = k / n_fill
            push!(βs, interpolate_power(x, y, t, q))
        end
    end

    push!(βs, βgeo[end])
    return βs
end

function insert_arithmetic_points(a::AbstractVector, n_insert::Int; start_idx::Int=1)
    n_insert >= 0 || error("--tail-insert must be nonnegative.")

    N = length(a)
    N >= 1 || error("array must be nonempty.")
    1 <= start_idx <= N || error("start_idx must satisfy 1 <= start_idx <= length(a).")

    n_insert_intervals = max(0, N - start_idx)
    out_len = N + n_insert * n_insert_intervals
    b = similar(a, out_len)

    idx = 1

    for i in 1:N-1
        x, y = a[i], a[i + 1]

        b[idx] = x
        idx += 1

        if i >= start_idx
            for k in 1:n_insert
                t = k / (n_insert + 1)
                b[idx] = (1 - t) * x + t * y
                idx += 1
            end
        end
    end

    b[idx] = a[end]
    return b
end

function log_power_ladder(β_min, β_max, n_geo, p; β_floor=0.1, eps_shift = 1e-5)
    β_min >= β_floor || error("β_min must be >= β_floor.")
    β_max >= β_min || error("β_max must be >= β_min.")
    n_geo >= 2 || error("n must be at least 2.")

    u_min = eps_shift + log(float(β_min) / β_floor)
    u_max = eps_shift + log(float(β_max) / β_floor)


    return [β_floor * exp(interpolate_power(u_min, u_max, t, p) - eps_shift) for t in range(0, 1, length=n_geo)]
end

function print_json_betas(βs)
    println("\"betas\": [")
    for (i, β) in enumerate(βs)
        comma = i < length(βs) ? "," : ""
        println("    ", round(β, sigdigits=8), comma)
    end
    println("]")
end

function main(args=ARGS)
    β_min = get_arg(args, "min", 0.1)
    β_max = get_arg(args, "max", 300.0)

    n_geo = get_arg(args, "n-geo", 9)
#    n_fill = get_arg(args, "n-fill", 6)
#    q = get_arg(args, "q", 1.0)
    p = get_arg(args, "p", 1.0)

    tail_insert = get_arg(args, "tail-insert", 0)
    tail_start_offset = get_arg(args, "tail-start-offset", 0)

#    βs = geo_skeleton_power_fill(
#        β_min,
#        β_max;
#        n_geo=n_geo,
#        n_fill=n_fill,
#        q=q,
#    )

    βs = log_power_ladder(β_min, β_max, n_geo, p)

    if tail_insert > 0
        start_idx = length(βs) - tail_start_offset
        βs = insert_arithmetic_points(βs, tail_insert; start_idx=start_idx)
    end

    println(length(βs))
    print_json_betas(βs)
end

main()
