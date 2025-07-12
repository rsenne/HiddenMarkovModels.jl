"""
    AbstractHSMM

Abstract supertype for an HSMM amenable to simulation, inference and learning.

# Interface

To create your own subtype of `AbstractHSMM`, you need to implement the following methods:

- [`initialization`](@ref)
- [`transition_matrix`](@ref)
- [`obs_distributions`](@ref)
- [`duration_distributions`](@ref)
- [`fit!`](@ref) (for learning)

# Applicable functions

Any `AbstractHSMM` which satisfies the interface can be given to the following functions:

- [`rand`](@ref)
- [`logdensityof`](@ref)
- [`forward`](@ref)
- [`viterbi`](@ref)
- [`forward_backward`](@ref)
- [`baum_welch`](@ref) (if `[fit!](@ref)` is implemented)
"""
abstract type AbstractHSMM <: AbstractHMM end

"""
    duration_distributions(hsmm)
    duration_distributions(hsmm, control)

Return a vector of duration distributions, one for each state of `hsmm`.

These distribution objects should implement:
- `Distributions.pdf(dist, duration::Int)` for discrete duration probabilities
- `Distributions.ccdf(dist, duration::Int)` for survival probabilities  
- `StatsAPI.fit!(dist, duration_seq, weight_seq)` for learning
"""
function duration_distributions end

# Fallbacks for no control
duration_distributions(hsmm::AbstractHSMM, ::Nothing) = duration_distributions(hsmm)

function Base.eltype(hsmm::AbstractHSMM, obs, control)
    init_type = eltype(initialization(hsmm))
    trans_type = eltype(transition_matrix(hsmm, control))
    dist = obs_distributions(hsmm, control)[1]
    logdensity_type = typeof(logdensityof(dist, obs))
    return promote_type(init_type, trans_type, logdensity_type)
end

## HSMM-specific sampling (override the AbstractHMM version)

"""
    rand([rng,] hsmm::AbstractHSMM, T)
    rand([rng,] hsmm::AbstractHSMM, control_seq)

Simulate `hsmm` for `T` time steps, or when the sequence `control_seq` is applied.
Explicitly models state durations instead of geometric sojourn times.
    
Return a named tuple `(; state_seq, obs_seq, duration_seq)`.
"""
function Random.rand(rng::AbstractRNG, hsmm::AbstractHSMM, control_seq::AbstractVector)
    T = length(control_seq)
    N = length(hsmm)
    
    # Start in initial state
    init = initialization(hsmm)
    dummy_log_probas = fill(-Inf, N)
    current_state = rand(rng, LightCategorical(init, dummy_log_probas))
    
    # Generate first observation to determine type, then initialize properly-typed vectors
    obs_dists = obs_distributions(hsmm, control_seq[1])
    first_obs = rand(rng, obs_dists[current_state])
    
    # Initialize sequences with proper types
    state_seq = Vector{Int}(undef, T)
    obs_seq = Vector{typeof(first_obs)}(undef, T)  # Use concrete type!
    duration_seq = Vector{Int}(undef, T)
    
    t = 1
    while t <= T
        # Sample duration for current state
        duration_dists = duration_distributions(hsmm, control_seq[t])
        current_duration = rand(rng, duration_dists[current_state])
        
        # Fill in state sequence for this duration
        end_time = min(t + current_duration - 1, T)
        for τ in t:end_time
            state_seq[τ] = current_state
            duration_seq[τ] = current_duration
            
            # Generate observation
            obs_dists = obs_distributions(hsmm, control_seq[τ])
            if τ == 1
                obs_seq[τ] = first_obs
            else
                obs_seq[τ] = rand(rng, obs_dists[current_state])
            end
        end
        
        t = end_time + 1
        
        if t <= T
            trans = transition_matrix(hsmm, control_seq[t-1])
            # Note: no self-transitions in HSMM
            next_state = rand(rng, LightCategorical(trans[current_state, :], dummy_log_probas))
            current_state = next_state
        end
    end
    
    return (; state_seq, obs_seq, duration_seq)
end

function Random.rand(hsmm::AbstractHSMM, control_seq::AbstractVector)
    return rand(default_rng(), hsmm, control_seq)
end

function Random.rand(rng::AbstractRNG, hsmm::AbstractHSMM, T::Integer)
    return rand(rng, hsmm, Fill(nothing, T))
end

function Random.rand(hsmm::AbstractHSMM, T::Integer)
    return rand(hsmm, Fill(nothing, T))
end