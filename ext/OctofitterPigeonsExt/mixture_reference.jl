# Pigeons interface for Octofitter.MixtureGaussianReference (src/mixture_reference.jl):
# a fixed mixture of diagonal Gaussians used as the variational reference of
# stabilized PT. The informal "variational" contract of Pigeons
# (src/variational/variational.jl) is: activate_variational, update_reference!,
# variational_recorder_builders, plus the log_potential contract (callable on the
# state) and sample_iid!. The gradient methods mirror Pigeons' GaussianReference
# and are only used by gradient-based explorers (AutoMALA, MALA, AAPS).
# The state is the flat linked parameter vector of the Octofitter target.

const _MixRef = Octofitter.MixtureGaussianReference

# Use the fixed mixture from round first_round + 1 on (the update happens between
# rounds). Before that the variational leg keeps the prior reference.
Pigeons.activate_variational(ref::_MixRef, iterators::Pigeons.Iterators) =
    iterators.round ≥ ref.first_round

# Fixed reference: nothing to record, nothing to refit.
Pigeons.variational_recorder_builders(::_MixRef) = []
Pigeons.update_reference!(reduced_recorders, ::_MixRef, state) = nothing

# Exact iid draw at the reference end of the variational leg.
function Pigeons.sample_iid!(ref::_MixRef, replica, shared)
    Octofitter._mixture_draw!(replica.rng, replica.state, ref)
    return nothing
end

# LogDensityProblems interface (the log density itself is the callable defined in src).
Pigeons.LogDensityProblems.logdensity(ref::_MixRef, x) = ref(x)
Pigeons.LogDensityProblems.dimension(ref::_MixRef) = Octofitter._mixture_dim(ref)
Pigeons.LogDensityProblems.capabilities(::Type{<:_MixRef}) =
    Pigeons.LogDensityProblems.LogDensityOrder{1}()

# Analytic gradient with a per-replica buffer, as for GaussianReference.
Pigeons.LogDensityProblemsAD.ADgradient(kind::Pigeons.ADTypes.AbstractADType, ref::_MixRef, replica::Pigeons.Replica) =
    Pigeons.BufferedAD(ref, replica.recorders.buffers)

function Pigeons.LogDensityProblems.logdensity_and_gradient(buffered::Pigeons.BufferedAD{<:_MixRef}, x)
    return Octofitter._mixture_logdensity_and_gradient!(buffered.buffer, buffered.enclosed, x)
end

# Skip Pigeons' generic AD-object cache for the reference, as Pigeons does for its own
# variational references (src/variational/VariationalReference.jl).
Pigeons.get_buffer(::Pigeons.Augmentation{<:Dict{Symbol, Pigeons.BufferedAD}}, ::Symbol,
                   kind, ref::_MixRef, replica::Pigeons.Replica) =
    Pigeons.LogDensityProblemsAD.ADgradient(kind, ref, replica)
