"""
$(TYPEDEF)

Hidden Semi-Markov Model implementation.

# Fields

$(TYPEDFIELDS)
"""
struct HSMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector,
    VDur<:AbstractVector,
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
} <: AbstractHSMM
    "initial state probabilities"
    init::V
    "state transition probabilities (excluding self-transitions)"
    trans::M
    "observation distributions"
    dists::VD
    "state duration distributions"
    durations::VDur
    "logarithms of initial state probabilities"
    loginit::Vl
    "logarithms of state transition probabilities"
    logtrans::Ml

    function HSMM(
        init::AbstractVector,
        trans::AbstractMatrix,
        dists::AbstractVector,
        durations::AbstractVector,
    )
        # Remove self-transitions from trans matrix for HSMMs
        trans_no_self = copy(trans)
        for i in 1:size(trans, 1)
            trans_no_self[i, i] = 0.0
        end
        # Renormalize rows
        foreach(sum_to_one!, eachrow(trans_no_self))

        log_init = elementwise_log(init)
        log_trans = elementwise_log(trans_no_self)

        hsmm = new{
            typeof(init),
            typeof(trans_no_self),
            typeof(dists),
            typeof(durations),
            typeof(log_init),
            typeof(log_trans),
        }(
            init, trans_no_self, dists, durations, log_init, log_trans
        )
        @argcheck valid_hsmm(hsmm)
        return hsmm
    end
end

function Base.show(io::IO, hsmm::HSMM)
    return print(
        io,
        "Hidden Semi-Markov Model with:\n - initialization: $(hsmm.init)\n - transition matrix: $(hsmm.trans)\n - observation distributions: [$(join(hsmm.dists, ", "))]\n - duration distributions: [$(join(hsmm.durations, ", "))]",
    )
end

# Interface implementations
initialization(hsmm::HSMM) = hsmm.init
log_initialization(hsmm::HSMM) = hsmm.loginit
transition_matrix(hsmm::HSMM) = hsmm.trans
log_transition_matrix(hsmm::HSMM) = hsmm.logtrans
obs_distributions(hsmm::HSMM) = hsmm.dists
duration_distributions(hsmm::HSMM) = hsmm.durations

function StatsAPI.fit!(
    hsmm::HSMM,
    fb_storage::HSMMForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    (; γ, ξ, expected_durations) = fb_storage

    # UPDATE INITIAL STATE PROBABILITIES
    fill!(hsmm.init, zero(eltype(hsmm.init)))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        # Sum initial state probabilities
        for i in 1:length(hsmm)
            hsmm.init[i] += γ[i, t1]
        end
    end
    sum_to_one!(hsmm.init)

    # UPDATE TRANSITION PROBABILITIES
    fill!(hsmm.trans, zero(eltype(hsmm.trans)))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        # Sum transition marginals over time
        for t in t1:(t2 - 1)
            hsmm.trans .+= ξ[t]
        end
    end
    # Normalize rows (no self-transitions, so diagonal should remain 0)
    foreach(sum_to_one!, eachrow(hsmm.trans))

    # UPDATE OBSERVATION DISTRIBUTIONS
    for i in 1:length(hsmm)
        # Collect weights for all sequences
        weights = zeros(eltype(γ), length(obs_seq))
        for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            weights[t1:t2] .= view(γ, i, t1:t2)
        end

        # Only fit if we have meaningful weight
        total_weight = sum(weights)
        if total_weight > 1e-10
            weights_typed = Vector{Float64}(weights)
            fit_in_sequence!(hsmm.dists, i, obs_seq, weights_typed)
        end
    end

    # UPDATE DURATION DISTRIBUTIONS
    for i in 1:length(hsmm)
        max_duration = size(expected_durations, 2)

        # expected_durations[i,d] already contains the accumulated expected number
        # of times state i had duration d across all sequences
        weights = expected_durations[i, :]

        # Only fit if we have meaningful observations
        total_weight = sum(weights)
        if total_weight > 1e-10
            # Create arrays of durations and their weights
            durations_typed = Vector{Int}(collect(1:max_duration))
            weights_typed = Vector{Float64}(weights)

            # Fit the duration distribution with these weighted observations
            fit!(hsmm.durations[i], durations_typed, weights_typed)
        end
    end

    # === UPDATE LOG VERSIONS ===
    hsmm.loginit .= log.(hsmm.init)
    mynonzeros(hsmm.logtrans) .= log.(mynonzeros(hsmm.trans))

    # === SAFETY CHECK ===
    @argcheck valid_hsmm(hsmm)

    return nothing
end

# currently do not have control worked out
function StatsAPI.fit!(
    hsmm::HSMM,
    fb_storage::HSMMForwardBackwardStorage,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    return fit!(hsmm, fb_storage, obs_seq; seq_ends)
end
