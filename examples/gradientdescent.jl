# # Gradient Descent in HMMs

#= In this tutorial we will explore two ways one can use gradient descent when fitting HMMs. There are two ways one typically wants to use gradient descent in HMMs:
1. Fitting parameters of the observation model that do not have closed form updates, e.g., GLMs, neural networks, etc.
2. Fitting the entire HMM using gradient based optimization by usingg automatic differentiation.
We will explore both of these in this tutorial.
=# 
using DensityInterface
using Distributions
using HiddenMarkovModels
# using HMMTest  #src
using LinearAlgebra
using Optim
using Random
using StableRNGs
using StatsAPI
using Test  #src
rng = StableRNG(42);

#= For both parts of this tutorial we will use a simple HMM with Gaussian observations. Of course, using gradient descent in either of these examples in completely overkill, but it serves to illustrate the methods. 
Let's begin by creating a Normal distirebution model.=#

mutable struct NormalModel{T}
    μ::T
    logσ::T # unconstrained parameterization; σ = exp(logσ)
end

σ(mod::NormalModel) = exp(mod.logσ)

#= We have defined a simple probability model with two parameters: the mean and the log of the standard deviation. Using the log is intentional so that we can optimize over all real numbers without worrying about the positivity constraint on the standard deviation. But, we must now provide the common interface HiddenMarkovModels.jl requires (e.g., logdensityof, rand, fit!)=#
function DensityInterface.logdensityof(mod::NormalModel, obs::T) where T<:Real
    s = σ(mod)
    return -0.5 * log(2π) - log(s) - 0.5 * ((obs - mod.μ) / s)^2
end

DensityInterface.DensityKind(::NormalModel) = DensityInterface.HasDensity()

function Random.rand(rng::AbstractRNG, mod::NormalModel)
    return rand(rng, Normal(mod.μ, σ(mod)))
end

#= Because we are fitting a Gaussian, where the variance can collapse to zero, we will add some weak priors on the parameters to prevent this from happening. We will use a weak Normal prior on the mean and a prior on the log standard deviation that pulls it toward 1 with moderate strength. =#
const μ_prior     = Normal(0.0, 10.0)          # weak prior on mean
const logσ_prior  = Normal(log(1.0), 0.5)      # prior pulling σ toward ~1 with moderate strength

function StatsAPI.fit!(mod::NormalModel, data::AbstractVector{<:Real}, weights::AbstractVector{<:Real})

    function neglogpost(θ)
        μ, logσ = θ

        # likelihood term (weighted)
        nll = 0.0
        tmp = NormalModel(μ, logσ)
        for (y, w) in zip(data, weights)
            nll += -w * logdensityof(tmp, y)
        end

        # prior terms (MAP = maximize loglik + logprior  <=> minimize -loglik - logprior)
        nll += -logpdf(μ_prior, μ)
        nll += -logpdf(logσ_prior, logσ)

        return nll
    end

    θ0 = [mod.μ, mod.logσ]
    result = Optim.optimize(neglogpost, θ0, BFGS(); autodiff = :forward)
    mod.μ, mod.logσ = Optim.minimizer(result)
    return mod
end

#= Now that we have fully defined our observation model, we can create an HMM using it.=#
init_dist = [0.2, 0.7, 0.1]
init_trans = [0.9 0.05 0.05; 0.075 0.9 0.025; 0.1 0.1 0.8]
obs_dists = [NormalModel(-3.0, log(0.25)), NormalModel(0.0, log(0.5)), NormalModel(3.0, log(0.75))]

hmm_true = HMM(init_dist, init_trans, obs_dists);

#= We can now generate some data from this HMM.=#
state_seq, obs_seq = rand(rng, hmm_true, 10000);

#= Now we can fit a new HMM to this data using gradient descent to fit the observation model parameters. We will create a set of guesses for the initial parameters then fit the new model.=#
init_dist_guess = fill(1.0 / 3, 3)
init_trans_guess = [0.98 0.01 0.01; 0.01 0.98 0.01; 0.01 0.01 0.98]
obs_dist_guess = [NormalModel(-2.0, log(1.0)), NormalModel(2.0, log(1.0)), NormalModel(0.0, log(1.0))]

hmm_guess = HMM(init_dist_guess, init_trans_guess, obs_dist_guess)

hmm_est, lls = baum_welch(hmm_guess, obs_seq)

#= 
Great! We were able to fit the model using gradient descent within EM. Now let's explore how we can fit the entire HMM using gradient descent by leveraging automatic differentiation.
To do this, we will leverage the fact that forward algorithm of an HMM implicitly marginalizes the latent states, allowing us to compute the likelihood of the observations directly given the model parameters.
Thus, we can use the returned loglikelihood of the forward algorithm as our objective function to minimize (negative loglikelihood) with respect to all model parameters. It is worthwhile too think about this though,
each time we evaluate the objective function we must run the forward algorithm, which can be computationally expensive for large datasets. But, this method allows us to fit any parameterized HMM using gradient descent. But, we will get approximate
second-order information via BFGS to help speed up convergence.
=#

# stable softmax
function softmax(v::AbstractVector)
    m = maximum(v)
    ex = exp.(v .- m)
    ex ./ sum(ex)
end

function rowsoftmax(M::AbstractMatrix)
    K = size(M, 1)
    A = similar(M)
    for i in 1:K
        A[i, :] .= softmax(view(M, i, :))
    end
    return A
end

#= To use gradient descent on an HMM, we need to be able to respect the constraints of the model. Specifically, the initial state distribution and transition matrix must be valid probability distributions (i.e., non-negative and sum to one). 
We achieve this by parameterizing these components in an unconstrained space and then applying the softmax function to map them back to the probability simplex. This is achieved via the softmax function. We could have also used the approriate Lagrange multipliers.=# 
function unpack_to_hmm(θ::AbstractVector, K::Int)
    idx = 1

    ηπ = @view θ[idx:idx+K-1]; idx += K
    ηA = reshape(@view(θ[idx:idx+K*K-1]), K, K); idx += K*K
    μ  = @view θ[idx:idx+K-1]; idx += K
    logσ = @view θ[idx:idx+K-1]; idx += K

    π = softmax(ηπ)
    A = rowsoftmax(ηA)
    obs = [NormalModel(μ[k], logσ[k]) for k in 1:K]
    return HMM(π, A, obs)
end

# Convert a valid HMM into an unconstrained θ
function hmm_to_θ0(hmm::HMM)
    K = length(hmm.init)

    π = hmm.init
    A = hmm.trans

    ηπ = log.(π .+ eps())               # avoid -Inf
    ηA = log.(A .+ eps())               # same

    μ    = [hmm.dists[k].μ    for k in 1:K]
    logσ = [hmm.dists[k].logσ for k in 1:K]

    return vcat(ηπ, vec(ηA), μ, logσ)
end

function negloglik_from_θ(θ, obs_seq, K)
    hmm = unpack_to_hmm(θ, K)
    _, loglik = forward(hmm, obs_seq; error_if_not_finite=false)
    return -loglik[1]
end

K = 3
θ0 = hmm_to_θ0(hmm_guess)

obj(θ) = negloglik_from_θ(θ, obs_seq, K)

result = Optim.optimize(obj, θ0, BFGS(); autodiff = :forward)

hmm_est2 = unpack_to_hmm(result.minimizer, K);

#= We have now trained an HMM using gradient descent on all of the parameters! Let's quickly quick it matches the results we ggot fropm EM above.=#
isapprox(hmm_est.init, hmm_est2.init; atol=1e-3)
isapprox(hmm_est.trans, hmm_est2.trans; atol=1e-3)
for k in 1:K
    isapprox(hmm_est.dists[k].μ, hmm_est2.dists[k].μ; atol=1e-3)
    isapprox(σ(hmm_est.dists[k]), σ(hmm_est2.dists[k]); atol=1e-3)
end