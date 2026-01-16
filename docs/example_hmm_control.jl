using DensityInterface
using Random
using StatsAPI
using HiddenMarkovModels
using LinearAlgebra
import HiddenMarkovModels:
    initialization, log_initialization,
    transition_matrix, log_transition_matrix,
    obs_distributions
using Base.Threads: @threads

struct ControlledHMM{
    V<:AbstractVector,
    M<:AbstractMatrix,
    VD<:AbstractVector,
    Vl<:AbstractVector,
    Ml<:AbstractMatrix,
} <: HiddenMarkovModels.AbstractHMM
    init::V
    trans::M
    dists::VD
    loginit::Vl
    logtrans::Ml

    function ControlledHMM(init::AbstractVector, trans::AbstractMatrix, dists::AbstractVector)
        loginit  = HiddenMarkovModels.elementwise_log(init)
        logtrans = HiddenMarkovModels.elementwise_log(trans)
        hmm = new{typeof(init), typeof(trans), typeof(dists), typeof(loginit), typeof(logtrans)}(
            init, trans, dists, loginit, logtrans
        )
        HiddenMarkovModels.@argcheck HiddenMarkovModels.valid_hmm(hmm)
        return hmm
    end
end

HiddenMarkovModels.initialization(hmm::ControlledHMM) = hmm.init
HiddenMarkovModels.log_initialization(hmm::ControlledHMM) = hmm.loginit

initialization(hmm::ControlledHMM, control) = hmm.init
initialization(hmm::ControlledHMM, ::Nothing) = hmm.init
log_initialization(hmm::ControlledHMM, control) = hmm.loginit
log_initialization(hmm::ControlledHMM, ::Nothing) = hmm.loginit

HiddenMarkovModels.transition_matrix(hmm::ControlledHMM) = hmm.trans
HiddenMarkovModels.log_transition_matrix(hmm::ControlledHMM) = hmm.logtrans

transition_matrix(hmm::ControlledHMM, control) = hmm.trans
transition_matrix(hmm::ControlledHMM, ::Nothing) = hmm.trans
log_transition_matrix(hmm::ControlledHMM, control) = hmm.logtrans
log_transition_matrix(hmm::ControlledHMM, ::Nothing) = hmm.logtrans

HiddenMarkovModels.obs_distributions(hmm::ControlledHMM) = hmm.dists
obs_distributions(hmm::ControlledHMM, control) = hmm.dists
obs_distributions(hmm::ControlledHMM, ::Nothing) = hmm.dists

Base.length(hmm::ControlledHMM) = length(hmm.dists)


# (a) eltype must reflect control-aware emission logdensities
function Base.eltype(hmm::ControlledHMM, obs, control)
    init_type = eltype(HiddenMarkovModels.initialization(hmm))
    trans_type = eltype(HiddenMarkovModels.transition_matrix(hmm, control))
    dist = HiddenMarkovModels.obs_distributions(hmm, control)[1]
    logdensity_type = typeof(DensityInterface.logdensityof(dist, obs, control))
    return promote_type(init_type, trans_type, logdensity_type)
end

# (b) control-aware observation logdensities
function HiddenMarkovModels.obs_logdensities!(
    logb::AbstractVector{T},
    hmm::ControlledHMM,
    obs,
    control;
    error_if_not_finite::Bool=true,
) where {T}
    dists = HiddenMarkovModels.obs_distributions(hmm, control)
    @inbounds for i in eachindex(logb, dists)
        logb[i] = DensityInterface.logdensityof(dists[i], obs, control)
    end
    error_if_not_finite && HiddenMarkovModels.@argcheck maximum(logb) < typemax(T)
    return nothing
end

function HiddenMarkovModels.obs_logdensities!(logb::AbstractVector, hmm::ControlledHMM, obs, ::Nothing; kwargs...)
    throw(ArgumentError("ControlledHMM requires control at every time step for emission logdensity."))
end

# (c) control-aware sampling
function Random.rand(rng::AbstractRNG, hmm::ControlledHMM, control_seq::AbstractVector)
    T = length(control_seq)
    dummy_log_probas = fill(-Inf, length(hmm))

    init = HiddenMarkovModels.initialization(hmm)
    state_seq = Vector{Int}(undef, T)
    state1 = rand(rng, HiddenMarkovModels.LightCategorical(init, dummy_log_probas))
    state_seq[1] = state1

    @inbounds for t in 1:(T - 1)
        trans = HiddenMarkovModels.transition_matrix(hmm, control_seq[t + 1]) # ignores control by design
        state_seq[t + 1] = rand(rng, HiddenMarkovModels.LightCategorical(trans[state_seq[t], :], dummy_log_probas))
    end

    dists1 = HiddenMarkovModels.obs_distributions(hmm, control_seq[1])
    obs1 = rand(rng, dists1[state1], control_seq[1])
    obs_seq = Vector{typeof(obs1)}(undef, T)
    obs_seq[1] = obs1

    @inbounds for t in 2:T
        dists = HiddenMarkovModels.obs_distributions(hmm, control_seq[t])
        obs_seq[t] = rand(rng, dists[state_seq[t]], control_seq[t])
    end
    return (; state_seq, obs_seq)
end

function Random.rand(hmm::ControlledHMM, T::Integer)
    throw(ArgumentError("ControlledHMM requires a control sequence; call rand(hmm, control_seq)."))
end

# helper: controlled emission fitting
function HiddenMarkovModels.fit_in_sequence!(
    dists::AbstractVector, i::Integer, x, control_seq, w
)
    StatsAPI.fit!(dists[i], x, control_seq, w)
    return nothing
end

# controlled fit! for the HMM itself
function StatsAPI.fit!(
    hmm::ControlledHMM,
    fb_storage::HiddenMarkovModels.ForwardBackwardStorage,
    obs_seq::AbstractVector,
    control_seq::AbstractVector;
    seq_ends::HiddenMarkovModels.AbstractVectorOrNTuple{Int},
)
    (; γ, ξ) = fb_storage

    # Same state fitting logic as HMM.fit!
    if seq_ends isa NTuple
        for k in eachindex(seq_ends)
            t1, t2 = HiddenMarkovModels.seq_limits(seq_ends, k)
            scratch = ξ[t2]
            fill!(scratch, zero(eltype(scratch)))
            for t in t1:(t2 - 1)
                scratch .+= ξ[t]
            end
        end
    else
        @threads for k in eachindex(seq_ends)
            t1, t2 = HiddenMarkovModels.seq_limits(seq_ends, k)
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
        t1, t2 = HiddenMarkovModels.seq_limits(seq_ends, k)
        hmm.init .+= view(γ, :, t1)
        hmm.trans .+= ξ[t2]
    end
    HiddenMarkovModels.sum_to_one!(hmm.init)
    foreach(HiddenMarkovModels.sum_to_one!, eachrow(hmm.trans))

    # Fit emissions with control
    for i in 1:length(hmm)
        HiddenMarkovModels.fit_in_sequence!(hmm.dists, i, obs_seq, control_seq, view(γ, i, :))
    end

    # Update logs
    hmm.loginit .= log.(hmm.init)
    HiddenMarkovModels.mynonzeros(hmm.logtrans) .= log.(HiddenMarkovModels.mynonzeros(hmm.trans))

    HiddenMarkovModels.@argcheck HiddenMarkovModels.valid_hmm(hmm)
    return nothing
end

function StatsAPI.fit!(
    hmm::ControlledHMM,
    fb_storage::HiddenMarkovModels.ForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::HiddenMarkovModels.AbstractVectorOrNTuple{Int},
)
    throw(ArgumentError("ControlledHMM requires control_seq; call fit!(hmm, fb_storage, obs_seq, control_seq; seq_ends=...)."))
end

function StatsAPI.fit!(
    hmm::ControlledHMM,
    fb_storage::HiddenMarkovModels.ForwardBackwardStorage,
    obs_seq::AbstractVector;
    control_seq=nothing,
    seq_ends::HiddenMarkovModels.AbstractVectorOrNTuple{Int},
)
    control_seq === nothing && throw(ArgumentError("ControlledHMM requires control_seq."))
    return StatsAPI.fit!(hmm, fb_storage, obs_seq, control_seq; seq_ends=seq_ends)
end


mutable struct GLMNormalModel{T}
    β0::T
    β1::T
    logσ::T
end

DensityInterface.DensityKind(::GLMNormalModel) = DensityInterface.HasDensity()

# Control-aware logdensity: accept any Real types and promote
function DensityInterface.logdensityof(mod::GLMNormalModel, obs::Real, control::Real)
    T = promote_type(typeof(mod.β0), typeof(obs), typeof(control))
    μ = T(mod.β0) + T(mod.β1) * T(control)
    s = exp(T(mod.logσ))
    z = (T(obs) - μ) / s
    return -T(0.5) * log(T(2) * T(pi)) - log(s) - T(0.5) * z * z
end

# Control-aware sampling: accept any Real control
function Random.rand(rng::AbstractRNG, mod::GLMNormalModel, control::Real)
    T = promote_type(typeof(mod.β0), typeof(control))
    μ = T(mod.β0) + T(mod.β1) * T(control)
    s = exp(T(mod.logσ))
    return μ + s * randn(rng, T)
end

# Adapter to match ControlledHMM's expected order
function StatsAPI.fit!(
    mod::GLMNormalModel,
    data::AbstractVector{<:Real},
    control_seq::AbstractVector{<:Real},
    weights::AbstractVector{<:Real},
)
    n = length(data)
    @assert length(control_seq) == n
    @assert length(weights) == n

    # Build X with consistent element type
    T = promote_type(typeof(mod.β0), eltype(data), eltype(control_seq), eltype(weights))
    X = Matrix{T}(undef, n, 2)
    @inbounds for i in 1:n
        X[i, 1] = one(T)
        X[i, 2] = T(control_seq[i])
    end
    y = T.(data)
    w = T.(weights)

    # Weighted normal equations
    W = Diagonal(w)
    β_est = (X' * W * X) \ (X' * W * y)

    residuals = y .- X * β_est
    σ_est = sqrt(sum(w .* (residuals .^ 2)) / sum(w))

    mod.β0 = β_est[1]
    mod.β1 = β_est[2]
    mod.logσ = log(σ_est)
    return nothing
end

#=
Example usage
=#
true_init = [0.5, 0.5]
true_trans = [0.8 0.2; 0.3 0.7]
true_dists = [GLMNormalModel(0.0, 1.0, log(1.0)), GLMNormalModel(5.0, -1.0, log(2.0))]
hmm = ControlledHMM(true_init, true_trans, true_dists)

control = collect(0.0:9.0)
sample = rand(hmm, control)
x = sample.obs_seq

hmm_est, _ = baum_welch(hmm, x, control)
