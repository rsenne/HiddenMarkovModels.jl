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
struct HSMMForwardStorage{R}
    "forward probabilities α[i,d,t] = P(X[t]=i, D[t]=d | Y[1:t])"
    α::Array{R,3}  # states × max_duration × time
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
struct HSMMForwardBackwardStorage{R}
    "posterior state marginals γ[i,t] = P(X[t]=i | Y[1:T])"
    γ::Matrix{R}
    "posterior duration marginals δ[i,d,t] = P(X[t]=i, D[t]=d | Y[1:T])"
    δ::Array{R,3}  # states × max_duration × time
    "one loglikelihood per observation sequence"
    logL::Vector{R}
    # Forward quantities
    α::Array{R,3}
    B::Matrix{R}
    c::Vector{R}
    # Backward quantities  
    β::Array{R,3}
    max_duration::Int
end

Base.eltype(::HSMMForwardBackwardStorage{R}) where {R} = R

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
    
    α = Array{R,3}(undef, N, max_duration, T)
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    
    return HSMMForwardStorage(α, logL, B, c, max_duration)
end

function _forward!(
    storage::HSMMForwardStorage,
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    error_if_not_finite::Bool = true,
)
    (; α, B, c, logL, max_duration) = storage
    t1, t2 = seq_limits(seq_ends, k)
    N = length(hsmm)
    
    logL[k] = zero(eltype(logL))
    
    for t in t1:t2
        # Compute observation likelihoods in log space for stability
        Bₜ = view(B, :, t)
        obs_logdensities!(Bₜ, hsmm, obs_seq[t], control_seq[t]; error_if_not_finite)
        
        # Convert to probabilities with max normalization
        logmax = maximum(Bₜ)
        Bₜ .= exp.(Bₜ .- logmax)
        
        if t == t1
            # === INITIALIZATION ===
            init = initialization(hsmm)
            for i in 1:N
                α[i, 1, t] = init[i] * Bₜ[i]
                # Initialize all other durations to zero
                for d in 2:max_duration
                    α[i, d, t] = zero(eltype(α))
                end
            end
        else
            # === FORWARD RECURSION ===
            αₜ = view(α, :, :, t)
            αₜ₋₁ = view(α, :, :, t-1)
            fill!(αₜ, zero(eltype(α)))
            
            # Get model parameters for this time step
            # Note: control at time t-1 affects transition from t-1 to t
            trans = transition_matrix(hsmm, control_seq[t-1]) 
            durations = duration_distributions(hsmm, control_seq[t])
            
            for i in 1:N
                for d in 1:max_duration
                    
                    # === CASE 1: CONTINUE IN SAME STATE ===
                    # We were in state i with duration d-1 at time t-1
                    # and we continue to stay in state i
                    if d > 1
                        # Probability of continuing: P(Duration ≥ d | Duration ≥ d-1)
                        ccdf_d_minus_1 = ccdf(durations[i], d-1)
                        ccdf_d = ccdf(durations[i], d)
                        
                        if ccdf_d_minus_1 > 0
                            continue_prob = ccdf_d / ccdf_d_minus_1
                            α[i, d, t] += αₜ₋₁[i, d-1] * continue_prob
                        end
                    end
                    
                    # === CASE 2: ENTER FROM ANOTHER STATE ===
                    # We transition into state i from some other state j
                    # This can only happen with duration d = 1 (just entered)
                    if d == 1
                        for j in 1:N
                            if i != j  # No self-transitions in HSMM
                                # Sum over all durations that could end at time t-1
                                enter_contribution = zero(eltype(α))
                                for d_prev in 1:max_duration
                                    # Probability that state j ends after exactly d_prev steps
                                    end_prob = pdf(durations[j], d_prev)
                                    enter_contribution += αₜ₋₁[j, d_prev] * end_prob
                                end
                                
                                # Apply transition probability
                                α[i, d, t] += enter_contribution * trans[j, i]
                            end
                        end
                    end
                    
                    # === APPLY OBSERVATION LIKELIHOOD ===
                    α[i, d, t] *= Bₜ[i]
                end
            end
        end
        
        # === NORMALIZATION ===
        total_prob = sum(α[:, :, t])
        if total_prob > 0
            c[t] = inv(total_prob)
            α[:, :, t] .*= c[t]
            logL[k] += -log(c[t]) + logmax  # Add back the log normalization
        else
            # Handle degenerate case
            c[t] = one(eltype(c))
            logL[k] += -Inf
        end
    end
    
    error_if_not_finite && @argcheck isfinite(logL[k])
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
    
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward!(storage, hsmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward!(storage, hsmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite)
        end
    end
    
    return storage.α, storage.logL
end
