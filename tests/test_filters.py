import numpy as np

from vision.filters import Kalman, OneEuro, Track, assign


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    f = OneEuro(min_cutoff=1.0, beta=0.02)
    raw = 300 + rng.normal(0, 3, (120, 2))
    out = np.array([f(x, i / 30) for i, x in enumerate(raw)])
    assert out[30:].std(axis=0).mean() < 0.5 * raw[30:].std(axis=0).mean()


def test_one_euro_follows_fast_motion():
    f = OneEuro(min_cutoff=1.0, beta=0.02)
    xs = [np.array([10.0 * i, 0.0]) for i in range(60)]
    out = [f(x, i / 30) for i, x in enumerate(xs)]
    assert abs(out[-1][0] - xs[-1][0]) < 25


def test_kalman_coasts_with_velocity():
    k = Kalman((0, 0), 2000, 4)
    for i in range(1, 20):
        k.predict(1 / 30)
        k.correct((10 * i, 5 * i))
    p = k.predict(1 / 30)
    assert abs(p[0] - 200) < 5 and abs(p[1] - 100) < 5
    p = k.predict(1 / 30)
    assert abs(p[0] - 210) < 8


def test_assign_prefers_more_matches_then_lower_cost():
    assert sorted(assign([[10, 50], [12, 60]], [100, 100])) == [(0, 1), (1, 0)] or \
        sorted(assign([[10, 50], [12, 60]], [100, 100])) == [(0, 0), (1, 1)]
    assert sorted(assign([[10, 20], [15, 90]], [100, 100])) == [(0, 1), (1, 0)]
    assert assign([[150]], [100]) == []
    assert assign([], []) == []


def test_identities_survive_crossing(cfg):
    tc = cfg["tracking"]
    tracks = [Track(0, (100, 200), 0.0, tc), Track(1, (500, 260), 0.0, tc)]
    for i in range(1, 60):
        t = i / 30
        for tr in tracks:
            tr.predict(1 / 30)
        a = (100 + 7 * i, 200 + i)
        b = (500 - 7 * i, 260 - i)
        cost = [[float(np.hypot(tr.pred[0] - x, tr.pred[1] - y)) for x, y in (a, b)] for tr in tracks]
        for ti, ci in assign(cost, [tc["gate"]] * 2):
            tracks[ti].update((a, b)[ci], t, 40, -1, 1.0, None)
    assert tracks[0].out[0] > 450 and tracks[1].out[0] < 150
