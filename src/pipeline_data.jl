module PipelineData

export EquilibrationData, SamplingData

struct EquilibrationData{S,R,C,A}
    snapshot::S
    result::R
    config::C
    adapt_success::Bool
    adapt_history::A
end

struct SamplingData{R,C}
    result::R
    config::C
end

end
