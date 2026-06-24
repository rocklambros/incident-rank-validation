from engine.model.sigma_u_sensitivity import is_prior_dominated, sweep_sigma_u


def test_sweep_collects_sigma_u_per_scale() -> None:
    calls = []

    def run_fn(scale: float) -> float:
        calls.append(scale)
        return 2.0  # data-dominated: same posterior regardless of prior

    out = sweep_sigma_u((0.5, 1.0, 2.0), run_fn)
    assert out == {0.5: 2.0, 1.0: 2.0, 2.0: 2.0}
    assert calls == [0.5, 1.0, 2.0]


def test_data_dominated_is_not_prior_dominated() -> None:
    # sigma_u stays ~2.0 across very different priors -> data identifies it.
    by_scale = {0.5: 2.0, 1.0: 2.0, 2.0: 2.05}
    assert is_prior_dominated((0.5, 1.0, 2.0), by_scale) is False


def test_prior_dominated_when_sigma_u_tracks_prior() -> None:
    # sigma_u ~ scale*const across priors -> posterior follows prior -> prior-dominated.
    by_scale = {0.5: 0.4, 1.0: 0.8, 2.0: 1.6}
    assert is_prior_dominated((0.5, 1.0, 2.0), by_scale) is True
