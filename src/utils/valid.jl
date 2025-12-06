function valid_prob_vec(p::AbstractVector{T}) where {T}
    return minimum(p) >= zero(T) && sum(p) ≈ one(T)
end

function valid_trans_mat(A::AbstractMatrix)
    return size(A, 1) == size(A, 2) && all(valid_prob_vec, eachrow(A))
end

function valid_dists(d::AbstractVector)
    for i in eachindex(d)
        if DensityKind(d[i]) == NoDensity()
            return false
        end
    end
    return true
end

"""
    valid_hmm(hmm)

Perform some checks to rule out obvious inconsistencies with an `AbstractHMM` object.
"""
function valid_hmm(hmm::AbstractHMM, control=nothing)
    init = initialization(hmm)
    trans = transition_matrix(hmm, control)
    dists = obs_distributions(hmm, control)
    if !(length(init) == length(dists) == size(trans, 1) == size(trans, 2))
        return false
    elseif !valid_prob_vec(init)
        return false
    elseif !valid_trans_mat(trans)
        return false
    elseif !valid_dists(dists)
        return false
    end
    return true
end

"""
    valid_hsmm(hsmm, control=nothing)

Perform validation checks specific to HSMMs.
"""
function valid_hsmm(hsmm::AbstractHSMM, control=nothing)
    # Basic HMM validation
    if !valid_hmm(hsmm, control)
        return false
    end

    # HSMM-specific checks
    init = initialization(hsmm)
    trans = transition_matrix(hsmm, control)
    dists = obs_distributions(hsmm, control)
    durations = duration_distributions(hsmm, control)

    # Check dimensions match
    if length(durations) != length(init)
        return false
    end

    # Check no self-transitions (HSMMs don't allow them)
    for i in 1:length(hsmm)
        if trans[i, i] > 1e-10  # Allow for numerical precision
            return false
        end
    end

    # Check duration distributions are valid
    for dur_dist in durations
        if DensityKind(dur_dist) == NoDensity()
            return false
        end
        # Could add more specific duration distribution checks here
        # e.g., support should be positive integers
    end

    return true
end
