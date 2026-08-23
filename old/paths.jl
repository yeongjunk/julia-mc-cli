export DIR_EQ, DIR_SAMPLE, DIR_ANALYSIS,
       FN_EQ_RAW, FN_EQ_META, FN_EQ_DIAGNOSTICS,
       FN_SAMPLE_RAW, FN_SAMPLE_META,
       EXT_RAW, EXT_META,
       getfilename
     
       

const DIR_EQ       = "eq"
const DIR_SAMPLE   = "sample"
const DIR_ANALYSIS = "analysis"

const FN_EQ_RAW         = "eq_raw"
const FN_EQ_META        = "eq_meta"
const FN_EQ_DIAGNOSTICS = "eq_diagnostics"

const FN_SAMPLE_RAW  = "sample_raw"
const FN_SAMPLE_META = "sample_meta"

const EXT_RAW  = ".jld2"
const EXT_META = ".csv"



getfilename(stem::AbstractString, tag::AbstractString, ext::AbstractString) = "$(stem)_$(tag)$(ext)"
