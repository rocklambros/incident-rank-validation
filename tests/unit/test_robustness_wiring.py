"""Plan 8a Task 6: robustness-spec wiring gate + heterogeneous SpecResult fields.

Tests the decide-time completeness gate and the optional non-kappa SpecResult
fields with fake data only — no NUTS is run here (real multi-spec execution
is exercised in Plan 8e).
"""
import pytest

from engine.decide.robustness_multiplicity import RobustnessSpread, SpecResult


def _spec(name, k=0.2):
    return SpecResult(spec_name=name, weighted_kappa_median=k,
                      weighted_kappa_ci=(0.0, 0.4), flags=())


def test_assert_robustness_complete_raises_when_declared_spec_missing():
    from engine.cli.pipeline import assert_robustness_complete

    class M:
        robustness_specs = ("poisson_flat", "hierarchical_pooling")

    spread = RobustnessSpread(primary=_spec("negative_binomial_per_stratum"),
                              robustness=(_spec("poisson_flat"),))  # missing hierarchical
    with pytest.raises(ValueError, match="hierarchical_pooling"):
        assert_robustness_complete(M(), spread)


def test_assert_robustness_complete_passes_when_all_declared_present():
    from engine.cli.pipeline import assert_robustness_complete

    class M:
        robustness_specs = ("poisson_flat",)

    spread = RobustnessSpread(primary=_spec("negative_binomial_per_stratum"),
                              robustness=(_spec("poisson_flat"),))
    assert_robustness_complete(M(), spread)  # no raise


def test_specresult_carries_optional_heterogeneous_fields():
    s = SpecResult(spec_name="hierarchical_pooling", weighted_kappa_median=0.2,
                   weighted_kappa_ci=(0.0, 0.4), flags=(), sigma_u=2.1,
                   extra_rankings={"incidence": ("LLM09", "LLM02")})
    assert s.sigma_u == 2.1
    assert s.extra_rankings["incidence"] == ("LLM09", "LLM02")


def test_specresult_optional_fields_default_to_none():
    s = _spec("poisson_flat")
    assert s.sigma_u is None
    assert s.extra_rankings is None
