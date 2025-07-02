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

# Interface methods for AbstractHSMM
"""
    duration_distributions(hsmm)
    duration_distributions(hsmm, control)

Return a vector of duration distributions, one for each state of `hsmm`.
"""
function duration_distributions end

# Implementations for HSMM
initialization(hsmm::HSMM) = hsmm.init
log_initialization(hsmm::HSMM) = hsmm.loginit
transition_matrix(hsmm::HSMM) = hsmm.trans
log_transition_matrix(hsmm::HSMM) = hsmm.logtrans
obs_distributions(hsmm::HSMM) = hsmm.dists
duration_distributions(hsmm::HSMM) = hsmm.durations

# Fallbacks for no control
duration_distributions(hsmm::AbstractHSMM, ::Nothing) = duration_distributions(hsmm)

