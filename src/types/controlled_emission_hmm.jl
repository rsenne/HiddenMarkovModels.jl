"""
$(TYPEDEF)

Implementation of a Controlled HMM where control variables only influence the emission models.

$(TYPEDFIELDS)
"""
struct ControlledHMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector,
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
} <: AbstractHMM
    init::V
    trans::M
    dists::VD
    loginit::Vl
    logtrans::Ml

    function ControlledHMM(
        init::AbstractVector, trans::AbstractMatrix, dists::AbstractVector
    )
        loginit = elementwise_log(init)
        logtrans = elementwise_log(trans)
        hmm = new{typeof(init),typeof(trans),typeof(dists),typeof(loginit),typeof(logtrans)}(
            init, trans, dists, loginit, logtrans
        )
        @argcheck valid_hmm(hmm)
        return hmm
    end
end

initialization(hmm::ControlledHMM) = hmm.init
log_initialization(hmm::ControlledHMM) = hmm.loginit

initialization(hmm::ControlledHMM, control) = hmm.init
initialization(hmm::ControlledHMM, ::Nothing) = hmm.init
log_initialization(hmm::ControlledHMM, control) = hmm.loginit
log_initialization(hmm::ControlledHMM, ::Nothing) = hmm.loginit

transition_matrix(hmm::ControlledHMM) = hmm.trans
log_transition_matrix(hmm::ControlledHMM) = hmm.logtrans

transition_matrix(hmm::ControlledHMM, control) = hmm.trans
transition_matrix(hmm::ControlledHMM, ::Nothing) = hmm.trans
log_transition_matrix(hmm::ControlledHMM, control) = hmm.logtrans
log_transition_matrix(hmm::ControlledHMM, ::Nothing) = hmm.logtrans

obs_distributions(hmm::ControlledHMM) = hmm.dists
obs_distributions(hmm::ControlledHMM, control) = hmm.dists
obs_distributions(hmm::ControlledHMM, ::Nothing) = hmm.dists

Base.length(hmm::ControlledHMM) = length(hmm.dists)

function Base.eltype(hmm::ControlledHMM, obs, control)
    init_type = eltype(initialization(hmm))
    trans_type = eltype(transition_matrix(hmm, control))
    dist = obs_distributions(hmm, control)[1]
    logdensity_type = typeof(DensityInterface.logdensityof(dist, obs, control))
    return promote_type(init_type, trans_type, logdensity_type)
end

function obs_logdensities!(
    logb::AbstractVector{T},
    hmm::ControlledHMM,
    obs,
    control;
    error_if_not_finite::Bool=true,
) where {T}
    dists = obs_distributions(hmm, control)
    for i in eachindex(logb, dists)
        logb[i] = DensityInterface.logdensityof(dists[i], obs, control)
    end
    error_if_not_finite && @argcheck maximum(logb) < typemax(T)
    return nothing
end

function obs_logdensities!(
    logb::AbstractVector, hmm::ControlledHMM, obs, ::Nothing; kwargs...
)
    throw(
        ArgumentError(
            "ControlledHMM requires control at every time step for emission logdensity."
        ),
    )
end

function Random.rand(rng::AbstractRNG, hmm::ControlledHMM, control_seq::AbstractVector)
    T = length(control_seq)
    dummy_log_probas = fill(-Inf, length(hmm))

    init = initialization(hmm)
    state_seq = Vector{Int}(undef, T)
    state1 = rand(rng, LightCategorical(init, dummy_log_probas))
    state_seq[1] = state1

    for t in 1:(T - 1)
        trans = transition_matrix(hmm, control_seq[t + 1]) # ignores control by design
        state_seq[t + 1] = rand(
            rng, LightCategorical(trans[state_seq[t], :], dummy_log_probas)
        )
    end

    dists1 = obs_distributions(hmm, control_seq[1])
    obs1 = rand(rng, dists1[state1], control_seq[1])
    obs_seq = Vector{typeof(obs1)}(undef, T)
    obs_seq[1] = obs1

    for t in 2:T
        dists = obs_distributions(hmm, control_seq[t])
        obs_seq[t] = rand(rng, dists[state_seq[t]], control_seq[t])
    end
    return (; state_seq, obs_seq)
end

function Random.rand(hmm::ControlledHMM, T::Integer)
    throw(
        ArgumentError(
            "ControlledHMM requires a control sequence; call rand(hmm, control_seq)."
        ),
    )
end

function StatsAPI.fit!(
    hmm::ControlledHMM,
    fb_storage::ForwardBackwardStorage,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    (; γ, ξ) = fb_storage

    # Same state fitting logic as HMM.fit!
    if seq_ends isa NTuple
        for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            scratch = ξ[t2]
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    else
        @threads for k in eachindex(seq_ends)
            t1, t2 = seq_limits(seq_ends, k)
            scratch = ξ[t2]
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    end

    fill!(hmm.init, zero(eltype(hmm.init)))
    fill!(hmm.trans, zero(eltype(hmm.trans)))
    for k in eachindex(seq_ends)
        t1, t2 = seq_limits(seq_ends, k)
        hmm.init .+= view(γ, :, t1)
        hmm.trans .+= ξ[t2]
    end
    sum_to_one!(hmm.init)
    foreach(sum_to_one!, eachrow(hmm.trans))

    # Fit emissions with control
    for i in 1:length(hmm)
        fit_in_sequence!(hmm.dists, i, obs_seq, control_seq, view(γ, i, :))
    end

    # Update logs
    hmm.loginit .= log.(hmm.init)
    mynonzeros(hmm.logtrans) .= log.(mynonzeros(hmm.trans))

    @argcheck valid_hmm(hmm)
    return nothing
end

function StatsAPI.fit!(
    hmm::ControlledHMM,
    fb_storage::ForwardBackwardStorage,
    obs_seq::AbstractVector;
    control_seq=nothing,
    seq_ends::AbstractVectorOrNTuple{Int},
)
    control_seq === nothing && throw(
        ArgumentError(
            "ControlledHMM requires control_seq; call fit!(hmm, fb_storage, obs_seq, control_seq; seq_ends=...).",
        ),
    )
    return StatsAPI.fit!(hmm, fb_storage, obs_seq, control_seq; seq_ends=seq_ends)
end

"""
$(SIGNATURES)

ControlledHMM overload: `control_seq` is required, and emissions are evaluated as control-aware
`logdensityof(dist, obs, control)`.
"""
function joint_logdensityof(
    hmm::ControlledHMM,
    obs_seq::AbstractVector,
    state_seq::AbstractVector,
    control_seq::AbstractVector;
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
            logL += DensityInterface.logdensityof(
                dists[state_seq[t]], obs_seq[t], control_seq[t]
            )
        end
    end

    return logL
end
