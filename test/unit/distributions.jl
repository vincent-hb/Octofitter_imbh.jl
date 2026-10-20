@testset "Special Prior Distributions" begin
    @testset "Sine Distribution" begin
        sine = Sine()
        @test Distributions.minimum(sine) ≈ 0.0 + eps()
        @test Distributions.maximum(sine) ≈ π - eps()
        @test Distributions.insupport(sine, π/2)
        @test !Distributions.insupport(sine, -0.1)
        @test !Distributions.insupport(sine, π + 0.1)

        # Test PDF shape - peak should be at π/2
        @test Distributions.pdf(sine, π/2) > Distributions.pdf(sine, 0.1)
        @test Distributions.pdf(sine, π/2) > Distributions.pdf(sine, π-0.1)
    end

    @testset "UniformCircular" begin
        uc = UniformCircular()
        @test uc.domain ≈ 2π

        uc_custom = UniformCircular(1.0)
        @test uc_custom.domain ≈ 1.0

        # Test in model context
        @test_nowarn Planet(
            name="b",
            basis=Visual{KepOrbit},
            observations=[],
            variables=@variables begin
                a ~ LogUniform(0.1, 100)
                e ~ Uniform(0, 0.99)
                i ~ Sine()
                ω ~ UniformCircular()
                Ω ~ UniformCircular(π)  # Custom domain
                θ ~ UniformCircular()
            end
        )
    end

    @testset "KDE Priors" begin
        samples = randn(1000) .+ 10  # Normal(10,1) samples
        kde = Octofitter.KDEDist(samples)

        @test kde isa Octofitter.KDEDist
        @test Distributions.minimum(kde) ≈ minimum(samples)
        @test Distributions.maximum(kde) ≈ maximum(samples)
        @test Distributions.insupport(kde, 10.0)
        @test !Distributions.insupport(kde, minimum(samples) - 1)

        # Test in model context
        @test_nowarn Planet(
            name="b",
            basis=Visual{KepOrbit},
            observations=[],
            variables=@variables begin
                a ~ kde
                e ~ Uniform(0, 0.99)
                i ~ Sine()
                ω ~ UniformCircular()
                Ω ~ UniformCircular()
                θ ~ UniformCircular()
                tp = θ_at_epoch_to_tperi(θ, 50000; M=system.M, e, a, i, ω, Ω)
            end
        )
    end
end

@testset "Observable-based Priors" begin
    astrom_data = PlanetRelAstromLikelihood(
        Table(epoch = [50000, 50100], ra = [100.0, 110.0], dec = [50.0, 55.0], σ_ra = [1.0, 1.0], σ_dec = [1.0, 1.0]),
        name="obs_prior_test"
    )

    obs_prior = ObsPriorAstromONeil2019(astrom_data)
    @test obs_prior isa ObsPriorAstromONeil2019
    @test obs_prior.wrapped_like === astrom_data

    # Test in model context with period prior
    @test_nowarn Planet(
        name="b",
        basis=Visual{KepOrbit},
        observations=[astrom_data, obs_prior],
        variables=@variables begin
            e ~ Uniform(0.0, 0.5)
            i ~ Sine()
            ω ~ UniformCircular()
            Ω ~ UniformCircular()
            P ~ LogUniform(0.1, 150)
            a = ∛(system.M * P^2)
            θ ~ UniformCircular()
            tp = θ_at_epoch_to_tperi(θ, 50000; M=system.M, e, a, i, ω, Ω)
        end
    )

    # Test subsetting
    subset = Octofitter.likeobj_from_epoch_subset(obs_prior, 1:1)
    @test subset isa ObsPriorAstromONeil2019
end

@testset "Observable-based prior is |J| to the first power" begin
    # O'Neil et al. 2019 eq. 36a.  The same model with and without the wrapper,
    # evaluated at the same parameters, must differ by exactly log|J| (this was
    # 2log|J| before 2026-10).  |J| is computed independently from eq. 33.
    epoch = 55927.0
    astrom = PlanetRelAstromObs(
        Table(epoch=[epoch], ra=[-260.0], dec=[-51.0], σ_ra=[0.5], σ_dec=[0.5], cor=[0.0]);
        name="pos")
    function build(obs)
        b = Planet(name="A", basis=Visual{KepOrbit}, observations=[obs],
            variables=@variables begin
                M = system.M
                P ~ Uniform(10, 2_000_000)
                a = cbrt(M * P^2)
                e ~ Uniform(0.0, 0.99)
                i ~ Sine()
                ω ~ UniformCircular()
                Ω ~ UniformCircular()
                θ ~ UniformCircular()
                tp = θ_at_epoch_to_tperi(θ, $epoch; a, e, i, ω, Ω, M)
            end)
        sys = System(name="oneil_exponent", companions=[b], observations=[],
            variables=@variables begin
                plx ~ truncated(Normal(0.1842, 0.0017), lower=0)
                M ~ Uniform(100, 100000)
            end)
        Octofitter.LogDensityModel(sys; verbosity=0)
    end
    m_plain = build(astrom)
    m_prior = build(ObsPriorAstromONeil2019(astrom))
    rng = Octofitter.Random.Xoshiro(3)   # Random is loaded by Octofitter, not by runtests.jl
    for _ in 1:4
        θ  = m_plain.sample_priors(rng)
        θt = m_plain.link(θ)
        term = m_prior.ℓπcallback(θt) - m_plain.ℓπcallback(θt)
        nt = m_plain.arr2nt(θ)
        pl = nt.planets.A
        orb = Visual{KepOrbit}(; a=pl.a, e=pl.e, i=pl.i, ω=pl.ω, Ω=pl.Ω, tp=pl.tp, M=nt.M, plx=nt.plx)
        sol = orbitsolve(orb, epoch)
        e, m, E = pl.e, meananom(sol), eccanom(sol)
        jac = abs(3m*(e+cos(E)) + 2*(-2+e^2+e*cos(E))*sin(E)) * cbrt(period(orb)/365.25) / sqrt(1-e^2)
        @test term ≈ log(jac) rtol=1e-6
    end
end
