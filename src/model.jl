abstract type AbstractModelConfig end

function model_dim end
function model_N0 end
function make_problem end

problem_factory(model) = () -> make_problem(model)
