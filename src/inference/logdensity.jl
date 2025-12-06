"""
$(SIGNATURES)

Run the forward algorithm to compute the loglikelihood of `obs_seq` for `hmm`, integrating over all possible state sequences.
"""
function DensityInterface.logdensityof(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    _, logL = forward(hmm, obs_seq, control_seq; seq_ends, error_if_not_finite=false)
    return sum(logL)
end

"""
$(SIGNATURES)

Run the forward algorithm to compute the the joint loglikelihood of `obs_seq` and `state_seq` for `hmm`.
"""
function joint_logdensityof(
    hmm::AbstractHMM,
    obs_seq::AbstractVector,
    state_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    R = eltype(hmm, obs_seq[1], control_seq[1])
    logL = zero(R)
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        # Initialization
        init = initialization(hmm)
        logL += log(init[state_seq[t1]])
        # Transitions
        for t in t1:(t2 - 1)
            trans = transition_matrix(hmm, control_seq[t + 1])
            logL += log(trans[state_seq[t], state_seq[t + 1]])
        end
        # Observations
        for t in t1:t2
            dists = obs_distributions(hmm, control_seq[t])
            logL += logdensityof(dists[state_seq[t]], obs_seq[t])
        end
    end
    return logL
end

"""
$(SIGNATURES)

Run the forward algorithm to compute the loglikelihood of `obs_seq` for `hsmm`, integrating over all possible state sequences and durations.
"""
function DensityInterface.logdensityof(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
    max_duration::Int=50,
)
    _, logL = forward(
        hsmm, obs_seq, control_seq; seq_ends, max_duration, error_if_not_finite=false
    )
    return sum(logL)
end

"""
$(SIGNATURES)

Run the forward algorithm to compute the joint loglikelihood of `obs_seq` and `state_seq` for `hsmm`, 
given the durations spent in each state.
"""
function joint_logdensityof(
    hsmm::AbstractHSMM,
    obs_seq::AbstractVector,
    state_seq::AbstractVector,
    duration_seq::AbstractVector,
    control_seq::AbstractVector=Fill(nothing, length(obs_seq));
    seq_ends::AbstractVectorOrNTuple{Int}=(length(obs_seq),),
)
    R = eltype(hsmm, obs_seq[1], control_seq[1])
    logL = zero(R)

    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)

        # Initialization
        init = initialization(hsmm)
        logL += log(init[state_seq[t1]])

        # Process the sequence
        t = t1
        while t <= t2
            current_state = state_seq[t]
            current_duration = duration_seq[t]

            # Add observation likelihood for this time step
            dists = obs_distributions(hsmm, control_seq[t])
            logL += logdensityof(dists[current_state], obs_seq[t])

            # If this is the start of a new state visit, add duration and transition likelihoods
            if t == t1 || state_seq[t] != state_seq[t - 1]
                # Add duration likelihood
                durations = duration_distributions(hsmm, control_seq[t])
                logL += logdensityof(durations[current_state], current_duration)

                # Add transition likelihood (if not the first state)
                if t > t1
                    trans = transition_matrix(hsmm, control_seq[t])
                    prev_state = state_seq[t - 1]
                    logL += log(trans[prev_state, current_state])
                end
            end

            t += 1
        end
    end

    return logL
end
