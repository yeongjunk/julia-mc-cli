using Profile

include("./run_eq.jl")

function profile_main(args)
    Profile.init(n=10^7, delay=0.001)
    Profile.clear()
    main(args)
    main(args)

    @profile main(args)

    println("\n========== Tree profile ==========\n")
    Profile.print(
        format=:tree,
        maxdepth=20,
        mincount=10,
        C=false,
    )

    return nothing
end

profile_main(ARGS)
