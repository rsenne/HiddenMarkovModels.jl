"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardStorage{R}
    "posterior last state marginals `α[i] = ℙ(X[T]=i | Y[1:T])`"
    α::Matrix{R}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    B::Matrix{R}
    c::Vector{R}
end

"""
$(TYPEDEF)

# Fields

Only the fields with a description are part of the public API.

$(TYPEDFIELDS)
"""
struct ForwardBackwardStorage{R,M<:AbstractMatrix{R}}
    "posterior state marginals `γ[i,t] = ℙ(X[t]=i | Y[1:T])`"
    γ::Matrix{R}
    "posterior transition marginals `ξ[t][i,j] = ℙ(X[t]=i, X[t+1]=j | Y[1:T])`"
    ξ::Vector{M}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    B::Matrix{R}
    α::Matrix{R}
    c::Vector{R}
    β::Matrix{R}
    Bβ::Matrix{R}
end

Base.eltype(::ForwardBackwardStorage{R}) where {R} = R

const ForwardOrForwardBackwardStorage{R} = Union{
    ForwardStorage{R},ForwardBackwardStorage{R}
}

"""
$(SIGNATURES)
"""
function initialize_forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    α = Matrix{R}(undef, N, T)
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    return ForwardStorage(α, logL, B, c)
end

function _forward_digest_observation!(
    current_state_marginals::AbstractVector{<:Real},
    current_obs_likelihoods::AbstractVector{<:Real},
    hmm::AbstractHMM,
    obs,
    control;
    error_if_not_finite::Bool,
)
    a, b = current_state_marginals, current_obs_likelihoods

    obs_logdensities!(b, hmm, obs, control; error_if_not_finite)
    logm = maximum(b)
    b .= exp.(b .- logm)

    a .*= b
    c = inv(sum(a))
    lmul!(c, a)

    logL = -log(c) + logm
    return c, logL
end

function _forward!(
    storage::ForwardOrForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    error_if_not_finite::Bool,
)
    (; α, B, c, logL) = storage
    t1, t2 = seq_limits(seq_ends, k)
    logL[k] = zero(eltype(logL))
    for t in t1:t2
        αₜ = view(α, :, t)
        Bₜ = view(B, :, t)
        if t == t1
            copyto!(αₜ, initialization(hmm))
        else
            αₜ₋₁ = view(α, :, t - 1)
            predict_next_state!(αₜ, hmm, αₜ₋₁, control_seq[t])
        end
        cₜ, logLₜ = _forward_digest_observation!(
            αₜ, Bₜ, hmm, obs_seq[t], control_seq[t]; error_if_not_finite
        )
        c[t] = cₜ
        logL[k] += logLₜ
    end

    error_if_not_finite && @argcheck isfinite(logL[k])
    return nothing
end

"""
$(SIGNATURES)
"""
function forward!(
    storage::ForwardOrForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    error_if_not_finite::Bool=true,
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward algorithm to infer the current state after sequence `obs_seq` for `hmm`.

Return a tuple `(storage.α, storage.logL)` where `storage` is of type [`ForwardStorage`](@ref).
"""
function forward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    error_if_not_finite::Bool=true,
)
    storage = initialize_forward(hmm, obs_seq, control_seq; seq_ends)
    forward!(storage, hmm, obs_seq, control_seq; seq_ends, error_if_not_finite)
    return storage.α, storage.logL
end

#=
HSMM Implementations
=#
"""
$(TYPEDEF)

Storage for HSMM forward algorithm results.

# Fields

$(TYPEDFIELDS)
"""
struct HSMMForwardStorage{R<:Real}
    "forward probabilities alphastarl[t,i] = P(state=i, duration starts | Y[1:t])"
    alphastarl::Matrix{R}
    "forward probabilities alphal[t,i] = P(state=i | Y[1:t])"  
    alphal::Matrix{R}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    "observation likelihoods B[i,t] = P(Y[t] | X[t]=i)"
    B::Matrix{R}
    "normalization constants"
    c::Vector{R}
    "maximum duration considered"
    max_duration::Int
end

"""
$(TYPEDEF)

Storage for HSMM forward-backward algorithm results.

# Fields

$(TYPEDFIELDS) 
"""
struct HSMMForwardBackwardStorage{R<:Real,M<:AbstractMatrix{R}}
    "posterior state marginals γ[i,t] = P(X[t]=i | Y[1:T])"
    γ::Matrix{R}
    "posterior transition marginals ξ[t][i,j] = P(transition from i to j at time t | Y[1:T])"
    ξ::Vector{M}
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    
    # Forward quantities
    "forward probabilities alphastarl[t,i]"
    alphastarl::Matrix{R}
    "forward probabilities alphal[t,i]"
    alphal::Matrix{R}
    "observation likelihoods B[i,t]" 
    B::Matrix{R}
    "normalization constants"
    c::Vector{R}
    
    # Backward quantities
    "backward probabilities betal[t,i]"
    betal::Matrix{R}
    "backward probabilities betastarl[t,i]"
    betastarl::Matrix{R}
    
    "maximum duration considered"
    max_duration::Int
end

Base.eltype(::HSMMForwardBackwardStorage{R}) where {R} = R

const HSMMForwardOrHSMMForwardBackwardStorage{R} = Union{
    HSMMForwardStorage{R}, HSMMForwardBackwardStorage{R}
}

"""
$(SIGNATURES)

Initialize storage for HSMM forward algorithm.
"""
function initialize_hsmm_forward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int = 50
)
    N, T, K = length(hsmm), length(obs_seq), length(seq_ends)
    R = eltype(hsmm, obs_seq[1], control_seq[1])
    
    alphastarl = Matrix{R}(undef, N, T)
    alphal = Matrix{R}(undef, N, T) 
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    
    return HSMMForwardStorage{R}(alphastarl, alphal, logL, B, c, max_duration)
end

# Helper functions for HSMMs

function cumulative_obs_potentials(storage, hsmm, obs_seq, control, t::Int, max_duration::Int)
    N, T = size(storage.B)  
    stop = min(T, t + max_duration - 1)
    
    # Return cumulative sum from t onwards
    cB = zeros(eltype(storage.B), stop - t + 1, N)
    
    for τ in 1:(stop - t + 1)
        time_idx = t + τ - 1
        for i in 1:N
            if τ == 1
                cB[τ, i] = storage.B[i, time_idx]
            else
                cB[τ, i] = cB[τ-1, i] + storage.B[i, time_idx]
            end
        end
    end
    
    return cB, 0.0  # offset
end

function reverse_cumulative_obs_potentials(storage, hsmm, obs_seq, control, t::Int, max_duration::Int)
    N, T = size(storage.B) 
    start = max(1, t - max_duration + 1)
    
    # Return reverse cumulative sum up to t
    length = t - start + 1
    cB = zeros(eltype(storage.B), length, N)
    
    # Build cumulative from start to t
    for τ in 1:length
        time_idx = start + τ - 1
        for i in 1:N
            if τ == 1
                cB[τ, i] = storage.B[i, time_idx]
            else
                cB[τ, i] = cB[τ-1, i] + storage.B[i, time_idx]
            end
        end
    end
    
    # Convert to reverse cumulative (PyHSMM style)
    total = cB[end, :]
    for τ in 1:length
        for i in 1:N
            cB[τ, i] = total[i] - (τ > 1 ? cB[τ-1, i] : 0.0)
        end
    end
    
    return cB
end

function dur_potentials(hsmm, t::Int, max_duration::Int, T::Int)
    durations = duration_distributions(hsmm)
    N = length(durations)
    stop = min(max_duration, T - t + 1)
    
    aDl = Matrix{Float64}(undef, stop, N)
    for d in 1:stop
        for i in 1:N
            aDl[d, i] = logpdf(durations[i], d)
        end
    end
    
    return aDl
end

function reverse_dur_potentials(hsmm, t::Int, max_duration::Int)
    durations = duration_distributions(hsmm)
    N = length(durations)
    stop = min(t, max_duration)
    
    aDl = Matrix{Float64}(undef, stop, N)
    for d in 1:stop
        for i in 1:N
            # This should be reversed to match PyHSMM: aDl[:stop][::-1]
            aDl[d, i] = logpdf(durations[i], stop - d + 1)
        end
    end
    
    return aDl
end

function dur_survival_potentials(hsmm, t::Int, max_duration::Int, T::Int)
    durations = duration_distributions(hsmm)
    N = length(durations)
    remaining_time = T - t + 1
    
    aDsl = fill(-Inf, N)
    if remaining_time <= max_duration
        for i in 1:N
            aDsl[i] = logccdf(durations[i], remaining_time)
        end
    end
    
    return aDsl
end

function _forward!(
    storage::HSMMForwardOrHSMMForwardBackwardStorage{R},
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    error_if_not_finite::Bool = true,
) where {R}
    
    t1, t2 = seq_limits(seq_ends, k)
    T = t2 - t1 + 1
    N = length(hsmm)
    max_duration = storage.max_duration
    
    # Pre-compute observation likelihoods
    for t in t1:t2
        obs_logdensities!(view(storage.B, :, t), hsmm, obs_seq[t], control_seq[t]; error_if_not_finite=false)
    end
    
    # Initialize alphastarl[1] = log(π₀)
    loginit = log_initialization(hsmm)
    for i in 1:N
        storage.alphastarl[t1, i] = loginit[i]
    end
    
    storage.logL[k] = zero(R)
    
    # Forward messages (following PyHSMM logic)
    for t in t1:(t2-1)
        t_rel = t - t1 + 1
        
        # Compute alphal[t] using reverse potentials
        cB = reverse_cumulative_obs_potentials(storage, hsmm, obs_seq, control_seq, t_rel, max_duration)
        rdp = reverse_dur_potentials(hsmm, t_rel, max_duration)
        
        # alphal[t] = logsumexp over durations
        for i in 1:N
            logsum_terms = R[]
            
            for τ in 1:min(size(cB, 1), size(rdp, 1), t_rel)
                start_time = t - τ + 1
                if start_time >= t1
                    alphastarl_val = storage.alphastarl[start_time, i]
                    term = alphastarl_val + cB[τ, i] + rdp[τ, i]
                    push!(logsum_terms, term)
                end
            end
            
            if !isempty(logsum_terms)
                storage.alphal[t, i] = logsumexp(logsum_terms)
            else
                storage.alphal[t, i] = -Inf
            end
        end
        
        # Compute alphastarl[t+1] from alphal[t] and transitions
        logtrans = log_transition_matrix(hsmm, control_seq[t+1])
        
        for j in 1:N
            logsum_terms = R[]
            for i in 1:N
                alphal_val = storage.alphal[t, i]
                term = alphal_val + logtrans[i, j]
                push!(logsum_terms, term)
            end
            
            if !isempty(logsum_terms)
                storage.alphastarl[t+1, j] = logsumexp(logsum_terms)
            else
                storage.alphastarl[t+1, j] = -Inf
            end
        end
    end
    
    # Final alphal[T]
    t = t2
    t_rel = T
    cB = reverse_cumulative_obs_potentials(storage, hsmm, obs_seq, control_seq, t_rel, max_duration)
    rdp = reverse_dur_potentials(hsmm, t_rel, max_duration)
    
    for i in 1:N
        logsum_terms = R[]
        
        for τ in 1:min(size(cB, 1), size(rdp, 1), t_rel)
            start_time = t - τ + 1
            if start_time >= t1
                alphastarl_val = storage.alphastarl[start_time, i]
                term = alphastarl_val + cB[τ, i] + rdp[τ, i]
                push!(logsum_terms, term)
            end
        end
        
        if !isempty(logsum_terms)
            storage.alphal[t, i] = logsumexp(logsum_terms)
        else
            storage.alphal[t, i] = -Inf
        end
    end
    
    # Compute normalizer
    logsum_terms = R[]
    for i in 1:N
        push!(logsum_terms, storage.alphal[t2, i])
    end
    
    storage.logL[k] = logsumexp(logsum_terms)
    
    error_if_not_finite && @argcheck isfinite(storage.logL[k])
    return nothing
end

"""
$(SIGNATURES)
"""
function forward!(
    storage::HSMMForwardOrHSMMForwardBackwardStorage,
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    error_if_not_finite::Bool=true,
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward!(storage, hsmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward!(storage, hsmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward algorithm to infer the current state after sequence `obs_seq` for `hsmm`.

Return a tuple `(storage.α, storage.logL)` where `storage` is of type [`HSMMForwardStorage`](@ref).
The forward probabilities are α[i,d,t] = P(X[t]=i, D[t]=d | Y[1:t]).
"""
function forward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=50,
    error_if_not_finite::Bool=true,
)
    storage = initialize_hsmm_forward(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    forward!(storage, hsmm, obs_seq, control_seq; seq_ends, error_if_not_finite)
    return storage.alphal, storage.logL
end