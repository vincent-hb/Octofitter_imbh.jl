# Tests for MixtureGaussianReference (src/mixture_reference.jl): the fixed mixture of
# diagonal Gaussians used as the variational reference of stabilized PT. The Pigeons
# methods live in ext/OctofitterPigeonsExt/mixture_reference.jl and are tested with a
# Pigeons run in Ocen_IMBH_analysis/tests/test_mixture_reference.jl (Pigeons is not a
# test dependency here).

using Random
using LinearAlgebra
using Statistics
using MCMCChains: Chains

@testset "MixtureGaussianReference" begin
    rng = Random.Xoshiro(11)
    D, K = 4, 2
    μ = [0.0 3.0; 1.0 -2.0; -1.0 0.5; 2.0 2.0]
    σ = [0.5 0.2; 1.0 0.3; 0.7 1.5; 0.1 0.4]
    w = [0.8, 0.2]
    ref = MixtureGaussianReference(μ, σ; weights=w)

    @testset "construction" begin
        @test Octofitter._mixture_dim(ref) == D
        @test Octofitter._mixture_ncomp(ref) == K
        @test sum(exp, ref.logweights) ≈ 1
        @test ref.first_round == 1
        # vector-of-vectors form
        ref2 = MixtureGaussianReference([μ[:, 1], μ[:, 2]], [σ[:, 1], σ[:, 2]]; weights=[4, 1])
        @test ref2.means == ref.means && ref2.stds == ref.stds
        @test ref2.logweights ≈ ref.logweights
        # equal weights by default
        @test MixtureGaussianReference(μ, σ).logweights ≈ log.([0.5, 0.5])
        @test_throws ErrorException MixtureGaussianReference(μ, σ[:, 1:1])
        @test_throws ErrorException MixtureGaussianReference(μ, -σ)
        @test_throws ErrorException MixtureGaussianReference(μ, σ; weights=[1.0])
        @test_throws ErrorException MixtureGaussianReference(μ, σ; weights=[1.0, 0.0])
        @test_throws ErrorException MixtureGaussianReference(μ, σ; first_round=0)
    end

    @testset "log density equals Distributions' mixture" begin
        dist = MixtureModel([MvNormal(μ[:, k], Diagonal(σ[:, k] .^ 2)) for k in 1:K], w)
        for _ in 1:20
            x = 3 .* randn(rng, D)
            @test ref(x) ≈ logpdf(dist, x) rtol = 1e-10
        end
        # far in the tail of both components: finite, no underflow to -Inf
        @test isfinite(ref(fill(1e3, D)))
        @test_throws ErrorException ref(zeros(D + 1))
    end

    @testset "gradient (central differences)" begin
        g = zeros(D)
        for _ in 1:10
            x = 2 .* randn(rng, D)
            l, g = Octofitter._mixture_logdensity_and_gradient!(g, ref, x)
            @test l ≈ ref(x)
            h = 1e-6
            gfd = map(1:D) do i
                e = zeros(D); e[i] = h
                (ref(x .+ e) - ref(x .- e)) / 2h
            end
            @test g ≈ gfd rtol = 1e-5 atol = 1e-7
        end
    end

    @testset "exact draws" begin
        N = 200_000
        X = zeros(D, N)
        x = zeros(D)
        for j in 1:N
            X[:, j] .= Octofitter._mixture_draw!(rng, x, ref)
        end
        # component membership via the 4th coordinate (means 2.0 vs 2.0 differ in σ,
        # so use the 1st: means 0 vs 3, σ 0.5 vs 0.2)
        f2 = mean(X[1, :] .> 2.0)
        @test abs(f2 - w[2]) < 5 * sqrt(w[2] * w[1] / N) + 1e-3
        # mixture mean and variance per coordinate
        m = μ * w
        v = (σ .^ 2 .+ μ .^ 2) * w .- m .^ 2
        @test all(abs.(vec(mean(X; dims=2)) .- m) .< 5 .* sqrt.(v ./ N))
        @test all(abs.(vec(var(X; dims=2)) ./ v .- 1) .< 0.02)
    end

    @testset "from chains: linked samples, name lookup, split" begin
        b = Planet(name="A", basis=Visual{KepOrbit}, observations=[],
            variables=@variables begin
                M = system.M
                P ~ Uniform(10, 1000)
                a = cbrt(M * P^2)
                e ~ Uniform(0.0, 0.99)
                i ~ Sine()
                ω ~ UniformCircular()
                Ω ~ UniformCircular()
                θ ~ UniformCircular()
                tp = θ_at_epoch_to_tperi(θ, 55927.0; a, e, i, ω, Ω, M)
            end)
        sys = System(name="mixref", companions=[b], observations=[],
            variables=@variables begin
                plx ~ truncated(Normal(0.1842, 0.0017), lower=0)
                M ~ LogUniform(100, 100000)
            end)
        model = Octofitter.LogDensityModel(sys; verbosity=0)
        names = Octofitter._flat_parameter_names(model.system)
        @test length(names) == model.D
        @test names == [:plx, :M, :A_P, :A_e, :A_i, :A_ωx, :A_ωy, :A_Ωx, :A_Ωy, :A_θx, :A_θy]

        N = 500
        Θ = reduce(hcat, [collect(model.sample_priors(rng)) for _ in 1:N])     # D × N, natural
        Θt = reduce(hcat, [collect(model.link(Θ[:, j])) for j in 1:N])         # D × N, linked
        # chain with the columns shuffled and an extra column, as in real chain files
        perm = randperm(rng, model.D)
        extra = randn(rng, N)
        chain = Chains(reshape(hcat(permutedims(Θ[perm, :]), extra), N, model.D + 1, 1),
                       vcat(names[perm], [:B_P]))
        X = Octofitter.linked_samples(model, chain)
        @test X ≈ Θt
        @test Octofitter.linked_samples(model, chain; ii=1:10) ≈ Θt[:, 1:10]

        # missing column -> error
        chain_missing = Chains(reshape(permutedims(Θ[2:end, :]), N, model.D - 1, 1), names[2:end])
        @test_throws ErrorException Octofitter.linked_samples(model, chain_missing)

        # one component per chain: moment-matched in linked space
        ref1 = mixture_reference_from_chains(model, chain)
        @test vec(ref1.means) ≈ vec(mean(Θt; dims=2))
        @test vec(ref1.stds) ≈ vec(std(Θt; dims=2))
        @test isfinite(ref1(Θt[:, 1]))

        # split on M: two components from one chain, scale applied to the stds
        cut = 3000.0
        ref2 = mixture_reference_from_chains(model, chain; split=(:M, cut), scale=1.5)
        lo = Θ[2, :] .< cut
        @test size(ref2.means, 2) == 2
        @test ref2.means[:, 1] ≈ vec(mean(Θt[:, lo]; dims=2))
        @test ref2.means[:, 2] ≈ vec(mean(Θt[:, .!lo]; dims=2))
        @test ref2.stds[:, 2] ≈ 1.5 .* vec(std(Θt[:, .!lo]; dims=2))

        # a component below min_draws is skipped
        ref3 = mixture_reference_from_chains(model, chain; split=(:M, 101.0), min_draws=50)
        @test size(ref3.means, 2) == 1

        # draws outside this model's support are dropped
        Θbad = copy(Θ); Θbad[3, 1:5] .= 5000.0                     # P > 1000
        chain_bad = Chains(reshape(permutedims(Θbad), N, model.D, 1), names)
        Xb = @test_logs (:warn,) Octofitter.linked_samples(model, chain_bad)
        @test size(Xb, 2) == N - 5
    end
end
