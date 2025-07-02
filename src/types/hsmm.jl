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

    function HSMM(init::AbstractVector, trans::AbstractMatrix, 
                  dists::AbstractVector, durations::AbstractVector)
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
            typeof(init),typeof(trans_no_self),typeof(dists),
            typeof(durations),typeof(log_init),typeof(log_trans)
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

## Fitting (will need to be implemented later for HSMM EM algorithm)

function StatsAPI.fit!(
    hsmm::HSMM,
    fb_storage::HSMMForwardBackwardStorage,
    obs_seq::AbstractVector;
    seq_ends::AbstractVectorOrNTuple{Int},
)
    # TODO: Implement HSMM-specific Baum-Welch
    # This is more complex than HMM fitting due to duration modeling
    error("HSMM fitting not yet implemented")
end