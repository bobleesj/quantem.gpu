"""``run(aberrations=...)`` applies every coefficient it names and says which higher orders it leaves applied.

An override replaces the named coefficients of the fit passed to ``run`` or, without one, of the session, which holds
what its latest fit, ``run()`` or assignment set. After ``run(aberrations=fifth_order)`` a mapping of C10_nm, C12_nm
and phi12_deg alone therefore reconstructs the fifth-order image again. It did so with no message: on a weak-phase
simulation the two calls differed by 0.0, two new sessions by 0.36 of the image standard deviation. The backend double
records the coefficients that reach it; no GPU is used.
"""

import pytest

from tests.contracts.test_direct_persistence import _Backend, _session

LOW = {"C10_nm": -5.0, "C12_nm": 2.0, "phi12_deg": 30.0}
FULL = {**LOW, "C21_nm": 40.0, "phi21_deg": 60.0, "C23_nm": 30.0, "phi23_deg": -20.0, "C30_nm": 3000.0,
        "C32_nm": 2000.0, "phi32_deg": 10.0, "C34_nm": 1500.0, "phi34_deg": -30.0, "C41_nm": 5e4, "phi41_deg": 70.0,
        "C43_nm": 4e4, "phi43_deg": -12.0, "C45_nm": 3e4, "phi45_deg": 3.0, "C50_nm": 3e5, "C52_nm": 4e5,
        "phi52_deg": 68.0, "C54_nm": 3e5, "phi54_deg": -3.0, "C56_nm": 3e5, "phi56_deg": -3.0}
# the 12 magnitudes beyond C12, in the engines\x27 slot order
HIGHER = [name for name in FULL if name.endswith("_nm") and name not in LOW]
KEPT = "stay applied"


class _Recording(_Backend):
    """The persistence double, keeping what each reconstruction received beyond C12.

    ``higher`` holds, per reconstruction, the 12 magnitudes of slots 2 to 13 in Angstrom, or None when the two-term
    kernel ran.
    """

    def __init__(self):
        super().__init__()
        self.higher = []

    def preview_upsampled(self, aberrations, **kwargs):
        magnitudes = kwargs.pop("higher_order_magnitudes", None)
        kwargs.pop("higher_order_angles", None)
        self.higher.append(None if magnitudes is None else [float(value) for value in magnitudes[2:]])
        return self.preview(aberrations, **kwargs)


@pytest.fixture
def new_session(tmp_path, monkeypatch):
    source = tmp_path / "scan.bin"
    source.write_bytes(b"detector counts")

    def new():
        backend = _Recording()
        return _session(monkeypatch, source, backend), backend

    return new


def test_every_named_coefficient_reaches_the_reconstruction(new_session, capsys):
    session, backend = new_session()
    result = session.run(aberrations=FULL, compute_loss=False)
    assert backend.higher[-1] == pytest.approx([10.0 * FULL[name] for name in HIGHER], rel=1e-6)
    assert result.aberrations == FULL
    session, backend = new_session()
    assert session.run(aberrations=LOW, compute_loss=False).aberrations == LOW
    assert backend.higher[-1] is None
    assert KEPT not in capsys.readouterr().out


def test_low_orders_after_a_fifth_order_run_keep_the_higher_orders_and_say_so(new_session, capsys):
    session, backend = new_session()
    session.run(aberrations=FULL, compute_loss=False)
    assert KEPT not in capsys.readouterr().out
    result = session.run(aberrations=LOW, compute_loss=False)
    said = capsys.readouterr().out
    assert backend.higher[-1] == backend.higher[-2] and backend.higher[-1] is not None
    assert result.aberrations == FULL
    assert f"12 higher-order terms the session holds from its latest fit, run() or assignment {KEPT}" in said
    assert ", ".join(HIGHER) in said
    assert "ptycho.known_aberrations" in said and "0.0" in said


def test_known_aberrations_or_named_zeros_reconstruct_without_the_higher_orders(new_session, capsys):
    session, backend = new_session()
    session.run(aberrations=FULL, compute_loss=False)
    session.known_aberrations = LOW
    assert session.run(compute_loss=False).aberrations == LOW
    assert backend.higher[-1] is None
    session.run(aberrations=FULL, compute_loss=False)
    session.run(aberrations={**LOW, **dict.fromkeys(HIGHER, 0.0)}, compute_loss=False)
    assert backend.higher[-1] is None
    assert KEPT not in capsys.readouterr().out


def test_an_override_of_a_fit_keeps_its_higher_orders_and_names_the_fit(new_session, capsys):
    fit = new_session()[0].run(aberrations=FULL, compute_loss=False)
    session, backend = new_session()
    result = session.run(fit, aberrations={"C10_nm": 12.5}, compute_loss=False)
    said = capsys.readouterr().out
    assert result.aberrations == {**FULL, "C10_nm": 12.5} and fit.aberrations == FULL
    assert backend.higher[-1] == pytest.approx([10.0 * FULL[name] for name in HIGHER], rel=1e-6)
    assert f"aberrations= names C10_nm; 12 higher-order terms of the fit passed to run() {KEPT}" in said
    assert "known_aberrations" not in said


def test_one_kept_term_reads_in_the_singular(new_session, capsys):
    session, _ = new_session()
    session.run(aberrations={**LOW, "C30_nm": 3000.0}, compute_loss=False)
    session.run(aberrations={"C10_nm": 1.0}, compute_loss=False)
    assert "1 higher-order term the session holds from its latest fit, run() or assignment stays applied (C30_nm)" \
        in capsys.readouterr().out


def test_a_low_order_override_without_higher_orders_prints_nothing(new_session, capsys):
    session, backend = new_session()
    session.run(aberrations=LOW, compute_loss=False)
    assert session.run(aberrations={"C10_nm": 1.0}, compute_loss=False).aberrations == {**LOW, "C10_nm": 1.0}
    assert backend.higher[-1] is None
    assert "aberrations= names" not in capsys.readouterr().out


@pytest.mark.parametrize("geometry", ({"sample_tilt_mrad": (2.0, -1.0)}, {"depth_spread_nm": 5.0}))
def test_a_sample_tilt_or_depth_key_says_where_they_are_given(new_session, geometry):
    session, backend = new_session()
    where = r"run\(fit\), or set ptycho\.sample_tilt_mrad = \(row, col\) and ptycho\.depth_spread_nm on the session"
    with pytest.raises(ValueError, match=where):
        session.run(aberrations={**LOW, **geometry}, compute_loss=False)
    with pytest.raises(ValueError, match=where):
        session.known_aberrations = {**LOW, **geometry}
    assert not backend.higher  # nothing was reconstructed


def test_an_unknown_coefficient_names_the_accepted_ones(new_session):
    session, _ = new_session()
    with pytest.raises(ValueError) as raised:
        session.run(aberrations={**LOW, "C21": 40.0}, compute_loss=False)
    message = str(raised.value)
    assert "unknown [\x27C21\x27]" in message
    assert "C21_nm, C23_nm, C30_nm" in message and "C56_nm" in message
    assert "tilt" not in message
