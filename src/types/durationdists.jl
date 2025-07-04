"""
Duration distributions for HSMMs with support on {1,2,3,...}.

These are thin wrappers around Distributions.jl types that provide the correct
support for HSMM duration modeling while maintaining a clean interface.
"""

using Distributions
using StatsAPI: fit!

## GeometricDuration - equivalent to HMM self-transitions

"""
    GeometricDuration{T} <: DiscreteUnivariateDistribution

Geometric duration distribution with support {1,2,3,...}.
Equivalent to Geometric(p) + 1 from Distributions.jl.

Parameters:
- `p`: success probability (probability of leaving the state each step)

Mean duration: 1/p
"""
mutable struct GeometricDuration{T<:Real} <: DiscreteUnivariateDistribution
    p::T
    
    function GeometricDuration(p::T) where T
        @argcheck 0 < p <= 1 "Success probability must be in (0,1]"
        return new{T}(p)
    end
end

Base.show(io::IO, d::GeometricDuration) = print(io, "GeometricDuration(p=$(d.p))")

# Distributions.jl interface
Distributions.pdf(d::GeometricDuration, k::Int) = k >= 1 ? d.p * (1 - d.p)^(k-1) : 0.0
Distributions.ccdf(d::GeometricDuration, k::Int) = k >= 0 ? (1 - d.p)^k : 1.0
Distributions.quantile(d::GeometricDuration, p::Real) = ceil(Int, log(1-p) / log(1-d.p))

Distributions.mean(d::GeometricDuration) = 1 / d.p
Distributions.var(d::GeometricDuration) = (1 - d.p) / d.p^2
Distributions.std(d::GeometricDuration) = sqrt(var(d))

function Base.rand(rng::AbstractRNG, d::GeometricDuration)
    return rand(rng, Geometric(d.p)) + 1
end

function StatsAPI.fit!(d::GeometricDuration{T}, durations::AbstractVector{Int}, weights::AbstractVector) where T
    weighted_mean = sum(durations .* weights) / sum(weights)
    new_p = 1 / weighted_mean
    # Modify in place to match package patterns
    d.p = clamp(new_p, 1e-10, 1.0)
    return nothing
end

## PoissonDuration - common choice for HSMMs

"""
    PoissonDuration{T} <: DiscreteUnivariateDistribution

Poisson duration distribution with support {1,2,3,...}.
Equivalent to Poisson(λ) + 1 from Distributions.jl.

Parameters:
- `λ`: rate parameter

Mean duration: λ + 1
"""
mutable struct PoissonDuration{T<:Real} <: DiscreteUnivariateDistribution
    λ::T
    
    function PoissonDuration(λ::T) where T
        @argcheck λ > 0 "Rate parameter must be positive"
        return new{T}(λ)
    end
end

Base.show(io::IO, d::PoissonDuration) = print(io, "PoissonDuration(λ=$(d.λ))")

function Distributions.pdf(d::PoissonDuration, k::Int)
    return k >= 1 ? pdf(Poisson(d.λ), k-1) : 0.0
end

function Distributions.ccdf(d::PoissonDuration, k::Int)
    return k >= 0 ? ccdf(Poisson(d.λ), k) : 1.0
end

function Distributions.quantile(d::PoissonDuration, p::Real)
    return quantile(Poisson(d.λ), p) + 1
end

Distributions.mean(d::PoissonDuration) = d.λ + 1
Distributions.var(d::PoissonDuration) = d.λ
Distributions.std(d::PoissonDuration) = sqrt(d.λ)

function Base.rand(rng::AbstractRNG, d::PoissonDuration)
    return rand(rng, Poisson(d.λ)) + 1
end

function StatsAPI.fit!(d::PoissonDuration{T}, durations::AbstractVector{Int}, weights::AbstractVector) where T
    weighted_mean = sum(durations .* weights) / sum(weights)
    new_λ = max(weighted_mean - 1, 1e-10)
    d.λ = new_λ
    return nothing
end

## NegBinomialDuration - for overdispersed durations

"""
    NegBinomialDuration{T} <: DiscreteUnivariateDistribution

Negative binomial duration distribution with support {1,2,3,...}.
Equivalent to NegativeBinomial(r, p) + 1 from Distributions.jl.

Parameters:
- `r`: number of successes
- `p`: success probability

Mean duration: r(1-p)/p + 1
Variance: r(1-p)/p²
"""
mutable struct NegBinomialDuration{T<:Real} <: DiscreteUnivariateDistribution
    r::T
    p::T
    
    function NegBinomialDuration(r::T, p::S) where {T,S}
        r_promoted, p_promoted = promote(r, p)
        @argcheck r_promoted > 0 "Number of successes must be positive"
        @argcheck 0 < p_promoted < 1 "Success probability must be in (0,1)"
        return new{typeof(r_promoted)}(r_promoted, p_promoted)
    end
end

Base.show(io::IO, d::NegBinomialDuration) = print(io, "NegBinomialDuration(r=$(d.r), p=$(d.p))")

function Distributions.pdf(d::NegBinomialDuration, k::Int)
    return k >= 1 ? pdf(NegativeBinomial(d.r, d.p), k-1) : 0.0
end

function Distributions.ccdf(d::NegBinomialDuration, k::Int)
    return k >= 0 ? ccdf(NegativeBinomial(d.r, d.p), k) : 1.0
end

function Distributions.quantile(d::NegBinomialDuration, p_quantile::Real)
    return quantile(NegativeBinomial(d.r, d.p), p_quantile) + 1
end

Distributions.mean(d::NegBinomialDuration) = d.r * (1 - d.p) / d.p + 1
Distributions.var(d::NegBinomialDuration) = d.r * (1 - d.p) / d.p^2
Distributions.std(d::NegBinomialDuration) = sqrt(var(d))

function Base.rand(rng::AbstractRNG, d::NegBinomialDuration)
    return rand(rng, NegativeBinomial(d.r, d.p)) + 1
end

function StatsAPI.fit!(d::NegBinomialDuration{T}, durations::AbstractVector{Int}, weights::AbstractVector) where T
    # Shift back to {0,1,2,...} for fitting
    shifted_durations = durations .- 1
    weighted_mean = sum(shifted_durations .* weights) / sum(weights)
    weighted_var = sum((shifted_durations .- weighted_mean).^2 .* weights) / sum(weights)
    
    if weighted_var > weighted_mean  # Overdispersed
        p_est = weighted_mean / weighted_var
        r_est = weighted_mean * p_est / (1 - p_est)
        
        d.p = clamp(p_est, 1e-10, 1 - 1e-10)
        d.r = max(r_est, 1e-10)
    else
        # Fall back to reasonable values for underdispersed case
        d.p = 0.5
        d.r = 2 * weighted_mean
    end
    
    return nothing
end