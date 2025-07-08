using Distributions
using HiddenMarkovModels
import HiddenMarkovModels as HMMs
using HiddenMarkovModels: LightDiagNormal, LightCategorical
using HMMTest
using LinearAlgebra
using Random: Random, AbstractRNG, default_rng, seed!
using SparseArrays
using StableRNGs
using Test

using HiddenMarkovModels: HSMM, GeometricDuration, PoissonDuration, NegBinomialDuration
using HiddenMarkovModels: AbstractHSMM, forward, valid_hsmm

TEST_SUITE = get(ENV, "JULIA_HMM_TEST_SUITE", "Standard")

## HSMM Settings

T, K = 100, 200

init = [0.4, 0.6]
init_guess = [0.5, 0.5]

# HSMM transition matrices have no self-transitions
trans = [0.0 1.0; 1.0 0.0]
trans_guess = [0.0 1.0; 1.0 0.0]

p = [[0.8, 0.2], [0.2, 0.8]]
p_guess = [[0.7, 0.3], [0.3, 0.7]]

μ = [-ones(2), +ones(2)]
μ_guess = [-0.8 * ones(2), +0.8 * ones(2)]

σ = ones(2)

rng = StableRNG(63)
control_seqs = [fill(nothing, rand(rng, T:(2T))) for k in 1:K]
control_seq = reduce(vcat, control_seqs)
seq_ends = cumsum(length.(control_seqs))

## Test functions for HSMMs

function test_hsmm_coherent_algorithms(rng, hsmm, control_seq; seq_ends, hsmm_guess=nothing, max_duration=50)
    """Test that HSMM algorithms produce coherent results"""
    
    # Test sampling
    sample_result = rand(rng, hsmm, control_seq)
    @test length(sample_result.state_seq) == length(control_seq)
    @test length(sample_result.obs_seq) == length(control_seq)
    @test length(sample_result.duration_seq) == length(control_seq)
    @test all(s -> s in 1:length(hsmm), sample_result.state_seq)
    @test all(d -> d >= 1, sample_result.duration_seq)
    
    # Test forward algorithm
    α, logL = forward(hsmm, sample_result.obs_seq, control_seq; seq_ends, max_duration)
    @test size(α) == (length(hsmm), max_duration, length(sample_result.obs_seq))
    @test length(logL) == length(seq_ends)
    @test all(isfinite, logL)
    
    # Test normalization
    for t in 1:length(sample_result.obs_seq)
        @test sum(α[:, :, t]) ≈ 1.0 atol=1e-10
    end
    
    # Test non-negativity
    @test all(α .>= 0)
    
    # Test impossible durations are zero
    for t in 1:length(sample_result.obs_seq)
        for d in (t+1):max_duration
            @test all(α[:, d, t] .< 1e-10)
        end
    end
    
    return true
end

function test_hsmm_type_stability(rng, hsmm, control_seq; seq_ends, hsmm_guess=nothing, max_duration=50)
    """Test that HSMM functions are type stable"""
    
    # Test forward algorithm type stability
    obs_seq = rand(rng, hsmm, control_seq).obs_seq
    @inferred forward(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    
    # Test sampling type stability
    @inferred rand(rng, hsmm, control_seq)
    @inferred rand(rng, hsmm, length(control_seq))
    
    return true
end

function test_hsmm_allocations(rng, hsmm, control_seq; seq_ends, hsmm_guess=nothing, max_duration=50)
    """Test that HSMM algorithms have reasonable allocations"""
    
    obs_seq = rand(rng, hsmm, control_seq).obs_seq
    
    # Pre-allocate storage
    storage = HMMs.initialize_hsmm_forward(hsmm, obs_seq, control_seq; seq_ends, max_duration)
    
    # Test that pre-allocated forward doesn't allocate much
    allocs = @allocated HMMs.forward!(storage, hsmm, obs_seq, control_seq; seq_ends)
    @test allocs < 1000  # Should be minimal allocations
    
    return true
end

## HSMM Tests

@testset verbose = true "HSMM Normal" begin
    dists = [Normal(μ[1][1]), Normal(μ[2][1])]
    dists_guess = [Normal(μ_guess[1][1]), Normal(μ_guess[2][1])]
    
    duration_dists = [GeometricDuration(0.3), GeometricDuration(0.4)]
    duration_dists_guess = [GeometricDuration(0.25), GeometricDuration(0.35)]

    hsmm = HSMM(init, trans, dists, duration_dists)
    hsmm_guess = HSMM(init_guess, trans_guess, dists_guess, duration_dists_guess)
    
    @test valid_hsmm(hsmm)
    @test valid_hsmm(hsmm_guess)

    rng = StableRNG(63)
    if TEST_SUITE == "Standard"
        test_hsmm_coherent_algorithms(rng, hsmm, control_seq; seq_ends, hsmm_guess)
        test_hsmm_type_stability(rng, hsmm, control_seq; seq_ends, hsmm_guess)
        test_hsmm_allocations(rng, hsmm, control_seq; seq_ends, hsmm_guess)
    end
end

@testset verbose = true "HSMM DiagNormal" begin
    dists = [MvNormal(μ[1], Diagonal(abs2.(σ))), MvNormal(μ[2], Diagonal(abs2.(σ)))]
    dists_guess = [
        MvNormal(μ_guess[1], Diagonal(abs2.(σ))), MvNormal(μ_guess[2], Diagonal(abs2.(σ)))
    ]
    
    duration_dists = [PoissonDuration(3.0), PoissonDuration(2.0)]
    duration_dists_guess = [PoissonDuration(2.5), PoissonDuration(1.5)]

    hsmm = HSMM(init, trans, dists, duration_dists)
    hsmm_guess = HSMM(init_guess, trans_guess, dists_guess, duration_dists_guess)
    
    @test valid_hsmm(hsmm)
    @test valid_hsmm(hsmm_guess)

    rng = StableRNG(63)
    if TEST_SUITE == "Standard"
        test_hsmm_coherent_algorithms(rng, hsmm, control_seq; seq_ends, hsmm_guess)
        test_hsmm_type_stability(rng, hsmm, control_seq; seq_ends, hsmm_guess)
    end
end

@testset verbose = true "HSMM LightCategorical" begin
    dists = [LightCategorical(p[1]), LightCategorical(p[2])]
    dists_guess = [LightCategorical(p_guess[1]), LightCategorical(p_guess[2])]
    
    duration_dists = [NegBinomialDuration(3.0, 0.4), NegBinomialDuration(2.0, 0.6)]
    duration_dists_guess = [NegBinomialDuration(2.5, 0.3), NegBinomialDuration(1.5, 0.5)]

    hsmm = HSMM(init, trans, dists, duration_dists)
    hsmm_guess = HSMM(init_guess, trans_guess, dists_guess, duration_dists_guess)
    
    @test valid_hsmm(hsmm)
    @test valid_hsmm(hsmm_guess)

    rng = StableRNG(63)
    if TEST_SUITE == "Standard"
        test_hsmm_coherent_algorithms(rng, hsmm, control_seq; seq_ends, hsmm_guess)
        test_hsmm_type_stability(rng, hsmm, control_seq; seq_ends, hsmm_guess)
        test_hsmm_allocations(rng, hsmm, control_seq; seq_ends, hsmm_guess)
    end
end

@testset verbose = true "HSMM LightDiagNormal" begin
    dists = [LightDiagNormal(μ[1], σ), LightDiagNormal(μ[2], σ)]
    dists_guess = [LightDiagNormal(μ_guess[1], σ), LightDiagNormal(μ_guess[2], σ)]
    
    duration_dists = [GeometricDuration(0.1), GeometricDuration(0.9)]
    duration_dists_guess = [GeometricDuration(0.15), GeometricDuration(0.85)]

    hsmm = HSMM(init, trans, dists, duration_dists)
    hsmm_guess = HSMM(init_guess, trans_guess, dists_guess, duration_dists_guess)
    
    @test valid_hsmm(hsmm)
    @test valid_hsmm(hsmm_guess)

    rng = StableRNG(63)
    if TEST_SUITE == "Standard"
        test_hsmm_coherent_algorithms(rng, hsmm, control_seq; seq_ends, hsmm_guess)
        test_hsmm_type_stability(rng, hsmm, control_seq; seq_ends, hsmm_guess)
        test_hsmm_allocations(rng, hsmm, control_seq; seq_ends, hsmm_guess)
    end
end

@testset verbose = true "HSMM Mixed Duration Types" begin
    dists = [Normal(μ[1][1]), Normal(μ[2][1])]
    dists_guess = [Normal(μ_guess[1][1]), Normal(μ_guess[2][1])]
    
    # Mix different duration distribution types
    duration_dists = [GeometricDuration(0.3), PoissonDuration(4.0)]
    duration_dists_guess = [GeometricDuration(0.25), PoissonDuration(3.0)]

    hsmm = HSMM(init, trans, dists, duration_dists)
    hsmm_guess = HSMM(init_guess, trans_guess, dists_guess, duration_dists_guess)
    
    @test valid_hsmm(hsmm)
    @test valid_hsmm(hsmm_guess)

    rng = StableRNG(63)
    if TEST_SUITE == "Standard"
        test_hsmm_coherent_algorithms(rng, hsmm, control_seq; seq_ends, hsmm_guess)
        test_hsmm_type_stability(rng, hsmm, control_seq; seq_ends, hsmm_guess)
    end
end

@testset verbose = true "HSMM Edge Cases" begin
    # Test with very short sequences
    short_control_seq = fill(nothing, 5)
    short_seq_ends = (5,)
    
    dists = [Normal(0.0), Normal(2.0)]
    duration_dists = [GeometricDuration(0.5), GeometricDuration(0.5)]
    hsmm = HSMM([0.5, 0.5], [0.0 1.0; 1.0 0.0], dists, duration_dists)
    
    rng = StableRNG(63)
    test_hsmm_coherent_algorithms(rng, hsmm, short_control_seq; seq_ends=short_seq_ends)
    
    # Test with single observation
    single_control_seq = fill(nothing, 1)
    single_seq_ends = (1,)
    test_hsmm_coherent_algorithms(rng, hsmm, single_control_seq; seq_ends=single_seq_ends)
    
    # Test with different max_duration settings
    for max_dur in [5, 10, 20]
        α, logL = forward(hsmm, [0.1, 0.2, 2.1], single_control_seq; seq_ends=(3,), max_duration=max_dur)
        @test size(α, 2) == max_dur
        @test isfinite(logL[1])
    end
end

@testset verbose = true "HSMM Numerical Stability" begin
    # Test with extreme parameters
    dists = [Normal(0.0, 0.01), Normal(100.0, 0.01)]  # Very different means
    duration_dists = [GeometricDuration(0.001), GeometricDuration(0.999)]  # Extreme durations
    hsmm = HSMM([0.999, 0.001], [0.0 1.0; 1.0 0.0], dists, duration_dists)
    
    rng = StableRNG(63)
    
    # Test that algorithms still work with extreme parameters
    obs_seq = [0.0, 0.0, 100.0, 100.0]
    control_seq = fill(nothing, 4)
    seq_ends = (4,)
    
    α, logL = forward(hsmm, obs_seq, control_seq; seq_ends)
    @test isfinite(logL[1])
    @test all(isfinite, α)
    @test all(α .>= 0)
    
    # Test sampling doesn't break
    sample_result = rand(rng, hsmm, 10)
    @test length(sample_result.state_seq) == 10
    @test all(s -> s in [1, 2], sample_result.state_seq)
end