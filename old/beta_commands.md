L=64:
#julia --project=../.. print_betas.jl   --min 0.1   --max 300   --n-geo 50   --n-fill 1   --q 0.6 --tail-insert 2   --tail-start-offset 3
julia --project=../.. print_betas.jl   --min 0.1   --max 300  --n-geo 50 --p 2.0 --tail-insert 2   --tail-start-offset 2

