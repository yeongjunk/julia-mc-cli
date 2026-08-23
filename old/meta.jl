using UUIDs
using Dates
using LibGit2




struct CommonMetadata
    run_id::String
    stage::String
    status::Bool
    created_at::String
    git_commit::String
    git_branch::String
    git_dirty::Bool
    julia_version::String
end

function CommonMetadata(
    stage::AbstractString;
    status::AbstractString = "running",
    git_commit::AbstractString,
    git_branch::AbstractString,
    git_dirty::Bool,
)
    return CommonMetadata(
        1,
        string(uuid4()),
        String(stage),
        String(status),
        string(now()),
        nothing,
        String(git_commit),
        String(git_branch),
        git_dirty,
        string(VERSION),
    )
end

function create_common_metadata(stage)
end
