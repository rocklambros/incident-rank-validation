from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult


def _spec(name: str, sigma_u: float | None = None) -> SpecResult:
    return SpecResult(
        spec_name=name, weighted_kappa_median=0.2,
        weighted_kappa_ci=(0.0, 0.4), flags=(), sigma_u=sigma_u,
    )


def test_render_shows_sigma_u_when_present() -> None:
    from engine.report.render import _render_robustness_lines

    spread = RobustnessSpread(
        primary=_spec("negative_binomial_per_stratum"),
        robustness=(_spec("hierarchical_pooling", sigma_u=2.19),),
    )
    text = "".join(_render_robustness_lines(spread))
    assert "hierarchical_pooling" in text
    assert "2.19" in text
    assert "σ_u" in text or "sigma_u" in text
