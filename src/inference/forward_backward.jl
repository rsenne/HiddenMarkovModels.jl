"""
$(SIGNATURES)
"""
function initialize_forward_backward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    transition_marginals=true,
)
    N, T, K = length(hmm), length(obs_seq), length(seq_ends)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    trans = transition_matrix(hmm, control_seq[1])
    M = typeof(similar(trans, R))

    γ = Matrix{R}(undef, N, T)
    ξ = Vector{M}(undef, T)
    if transition_marginals
        for t in 1:(T - 1)
            ξ[t] = similar(transition_matrix(hmm, control_seq[t + 1]), R)
        end
        ξ[T] = zero(trans)  # not used
    end
    logL = Vector{R}(undef, K)
    B = Matrix{R}(undef, N, T)
    α = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)
    β = Matrix{R}(undef, N, T)
    Bβ = Matrix{R}(undef, N, T)
    return ForwardBackwardStorage{R,M}(γ, ξ, logL, B, α, c, β, Bβ)
end

function _forward_backward!(
    storage::ForwardBackwardStorage{R},
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    transition_marginals::Bool=true,
) where {R}
    (; α, β, c, γ, ξ, B, Bβ) = storage
    t1, t2 = seq_limits(seq_ends, k)

    # Forward (fill B, α, c and logL)
    _forward!(storage, hmm, obs_seq, control_seq, seq_ends, k; error_if_not_finite=true)

    # Backward
    β[:, t2] .= c[t2]
    for t in (t2 - 1):-1:t1
        Bβ[:, t + 1] .= view(B, :, t + 1) .* view(β, :, t + 1)
        βₜ = view(β, :, t)
        Bβₜ₊₁ = view(Bβ, :, t + 1)
        predict_previous_state!(βₜ, hmm, Bβₜ₊₁, control_seq[t + 1])
        lmul!(c[t], βₜ)
    end
    Bβ[:, t1] .= view(B, :, t1) .* view(β, :, t1)

    # State marginals
    γ[:, t1:t2] .= view(α, :, t1:t2) .* view(β, :, t1:t2) ./ view(c, t1:t2)'

    # Transition marginals
    if transition_marginals
        for t in t1:(t2 - 1)
            trans = transition_matrix(hmm, control_seq[t + 1])
            mul_rows_cols!(ξ[t], view(α, :, t), trans, view(Bβ, :, t + 1))
        end
        ξ[t2] .= zero(R)
    end

    return nothing
end

"""
$(SIGNATURES)
"""
function forward_backward!(
    storage::ForwardBackwardStorage,
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    transition_marginals::Bool=true,
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward-backward algorithm to infer the posterior state and transition marginals during sequence `obs_seq` for `hmm`.

Return a tuple `(storage.γ, storage.logL)` where `storage` is of type [`ForwardBackwardStorage`](@ref).
"""
function forward_backward(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    transition_marginals = false
    storage = initialize_forward_backward(
        hmm, obs_seq, control_seq; seq_ends, transition_marginals
    )
    forward_backward!(storage, hmm, obs_seq, control_seq; seq_ends, transition_marginals)
    return storage.γ, storage.logL
end

#=
HSMM implementations
=#
"""
$(SIGNATURES)
"""
function initialize_hsmm_forward_backward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    max_duration::Int=50,
    transition_marginals::Bool=true,
)
    N, T, K = length(hsmm), length(obs_seq), length(seq_ends)
    R = eltype(hsmm, obs_seq[1], control_seq[1])
    trans = transition_matrix(hsmm, control_seq[1])
    M = typeof(similar(trans, R))

    γ = Matrix{R}(undef, N, T)
    ξ = Vector{M}(undef, T)
    if transition_marginals
        for t in 1:(T - 1)
            ξ[t] = similar(transition_matrix(hsmm, control_seq[t + 1]), R)
        end
        ξ[T] = zero(trans)
    end
    logL = Vector{R}(undef, K)

    # Forward quantities
    alphastarl = Matrix{R}(undef, N, T)
    alphal = Matrix{R}(undef, N, T)
    B = Matrix{R}(undef, N, T)
    c = Vector{R}(undef, T)

    # Backward quantities  
    betal = Matrix{R}(undef, N, T)
    betastarl = Matrix{R}(undef, N, T)

    expected_durations = Matrix{R}(undef, N, max_duration)

    # Work buffers
    cB_buffer = Matrix{R}(undef, max_duration, N)
    dp_buffer = Matrix{R}(undef, max_duration, N)
    surv_buffer = Vector{R}(undef, N)

    return HSMMForwardBackwardStorage{R,M}(
        γ,
        ξ,
        logL,
        alphastarl,
        alphal,
        B,
        c,
        betal,
        betastarl,
        expected_durations,
        max_duration,
        cB_buffer,
        dp_buffer,
        surv_buffer,
    )
end

function _compute_expected_durations!(
    storage::HSMMForwardBackwardStorage{R},
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer,
) where {R}
    t1, t2 = seq_limits(seq_ends, k)
    T = t2 - t1 + 1
    N = length(hsmm)
    max_duration = storage.max_duration

    # Use sequence log-likelihood as normalizer
    normalizer = storage.logL[k]

    logpmfs = fill(-Inf, N, max_duration)

    # Loop over time
    for t_rel in 1:T
        t_abs = t1 + t_rel - 1

        # Get potentials for this time step (in-place)
        cB_len, _ = cumulative_obs_potentials!(
            storage.cB_buffer, storage, hsmm, obs_seq, control_seq, t_rel, max_duration
        )
        dp_len = dur_potentials!(storage.dp_buffer, hsmm, t_rel, max_duration, T)

        for i in 1:N
            max_dur = min(max_duration, cB_len, dp_len, T - t_rel + 1)

            for d in 1:max_dur
                future_time_abs = t_abs + d - 1

                if future_time_abs <= t2
                    # Compute log P(state=i, duration=d starts at t | observations)
                    # = log P(reach state i at t) + log P(duration d) +
                    #   sum of log obs likelihoods + log P(future | end at t+d-1)
                    log_prob =
                        storage.alphastarl[i, t_abs] +
                        storage.dp_buffer[d, i] +
                        storage.cB_buffer[d, i] +
                        storage.betal[i, future_time_abs] - normalizer

                    # Accumulate in log-space
                    logpmfs[i, d] = logaddexp(logpmfs[i, d], log_prob)
                end
            end
        end
    end

    # Convert from log-space to expected counts
    # Note: We don't normalize here because we want expected COUNTS, not probabilities
    fill!(storage.expected_durations, zero(R))
    for i in 1:N
        for d in 1:max_duration
            if isfinite(logpmfs[i, d])
                storage.expected_durations[i, d] = exp(logpmfs[i, d])
            end
        end
    end

    return nothing
end

function _forward_backward!(
    storage::HSMMForwardBackwardStorage{R},
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector,
    seq_ends::AbstractVectorOrNTuple{Int},
    k::Integer;
    transition_marginals::Bool=true,
) where {R}
    t1, t2 = seq_limits(seq_ends, k)
    T = t2 - t1 + 1
    N = length(hsmm)
    max_duration = storage.max_duration

    # Run forward pass first
    _forward!(storage, hsmm, obs_seq, control_seq, seq_ends, k)

    # Backward pass using PyHSMM logic
    # Initialize betal[i, T] = 0
    for i in 1:N
        storage.betal[i, t2] = zero(R)
    end

    for t in t2:-1:t1
        t_rel = t - t1 + 1

        # Compute betastarl[i, t] using cumulative potentials (in-place)
        cB_len, offset = cumulative_obs_potentials!(
            storage.cB_buffer, storage, hsmm, obs_seq, control_seq, t_rel, max_duration
        )
        dp_len = dur_potentials!(storage.dp_buffer, hsmm, t_rel, max_duration, T)

        for i in 1:N
            betastarl_val = -Inf

            # Sum over possible durations
            for τ in 1:min(cB_len, dp_len)
                future_time = t + τ - 1
                if future_time <= t2
                    betal_val = storage.betal[i, future_time]
                    term = betal_val + storage.cB_buffer[τ, i] + storage.dp_buffer[τ, i]
                    betastarl_val = logaddexp(betastarl_val, term)
                end
            end

            if isfinite(betastarl_val)
                betastarl_val -= offset

                # Add right censoring if applicable
                if t + cB_len - 1 >= t2
                    dur_survival_potentials!(storage.surv_buffer, hsmm, t_rel, max_duration, T)
                    if isfinite(storage.surv_buffer[i])
                        censoring_term = storage.cB_buffer[cB_len, i] - offset + storage.surv_buffer[i]
                        betastarl_val = logaddexp(betastarl_val, censoring_term)
                    end
                end
            end

            storage.betastarl[i, t] = betastarl_val
        end

        # Compute betal[i, t-1] from betastarl[j, t]
        if t > t1
            logtrans = log_transition_matrix(hsmm, control_seq[t])

            for i in 1:N
                betal_val = -Inf
                for j in 1:N
                    betastarl_val = storage.betastarl[j, t]
                    term = betastarl_val + logtrans[i, j]
                    betal_val = logaddexp(betal_val, term)
                end
                storage.betal[i, t - 1] = betal_val
            end
        end
    end

    # Compute state marginals γ[i,t] = exp(alphal[i,t] + betal[i,t] - normalizer)
    for t in t1:t2
        # Compute normalizer
        normalizer = -Inf
        for i in 1:N
            alphal_val = storage.alphal[i, t]
            betal_val = storage.betal[i, t]
            term = alphal_val + betal_val
            normalizer = logaddexp(normalizer, term)
        end

        # Compute marginals
        for i in 1:N
            alphal_val = storage.alphal[i, t]
            betal_val = storage.betal[i, t]
            storage.γ[i, t] = exp(alphal_val + betal_val - normalizer)
        end
    end

    # Compute transition marginals if requested
    if transition_marginals
        for t in t1:(t2 - 1)
            fill!(storage.ξ[t], zero(R))
            logtrans = log_transition_matrix(hsmm, control_seq[t + 1])

            for i in 1:N
                for j in 1:N
                    # ξ[t][i,j] = exp(alphal[i,t] + logtrans[i,j] + betastarl[j,t+1] - normalizer)
                    alphal_val = storage.alphal[i, t]
                    betastarl_val = storage.betastarl[j, t + 1]
                    normalizer = storage.logL[k]  # Use sequence log-likelihood as normalizer

                    storage.ξ[t][i, j] = exp(
                        alphal_val + logtrans[i, j] + betastarl_val - normalizer
                    )
                end
            end
        end
        storage.ξ[t2] .= zero(R)
    end

    _compute_expected_durations!(storage, hsmm, obs_seq, control_seq, seq_ends, k)

    return nothing
end

"""
$(SIGNATURES)
"""
function forward_backward!(
    storage::HSMMForwardBackwardStorage,
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
    transition_marginals::Bool=true,
)
    if seq_ends isa NTuple{1}
        for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hsmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    else
        @threads for k in eachindex(seq_ends)
            _forward_backward!(
                storage, hsmm, obs_seq, control_seq, seq_ends, k; transition_marginals
            )
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Apply the forward-backward algorithm to infer the posterior state and transition marginals during sequence `obs_seq` for `hsmm`.

Return a tuple `(storage.γ, storage.logL)` where `storage` is of type [`HSMMForwardBackwardStorage`](@ref).
"""
function forward_backward(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=50,
)
    transition_marginals = false
    storage = initialize_hsmm_forward_backward(
        hsmm, obs_seq, control_seq; seq_ends, max_duration, transition_marginals
    )
    forward_backward!(storage, hsmm, obs_seq, control_seq; seq_ends, transition_marginals)
    return storage.γ, storage.logL
end
