"""Architecture tests for the backend-neutral SSB package."""

import inspect
import json
import math
from pathlib import Path


def test_public_ssb_api_has_one_workflow() -> None:
    """C1: scientist API, expect one session with explicit operations."""

    from quantem.gpu import SSB

    for name in (
        "open",
        "find_aberrations",
        "show_trials",
        "reconstruct",
        "preview",
        "close",
    ):
        assert hasattr(SSB, name)


def test_ssb_open_keeps_detector_storage_dtype_explicit() -> None:
    """C1b: raw HDF5 loading defaults native and permits an explicit dtype."""

    from quantem.gpu import SSB

    signature = inspect.signature(SSB.open)
    assert signature.parameters["dtype"].default is None


def test_ssb_operations_share_saved_result_controls() -> None:
    """C1b2: fit and fixed-probe reconstruction share persistence vocabulary."""

    from quantem.gpu import SSB

    for operation in (SSB.find_aberrations, SSB.reconstruct):
        signature = inspect.signature(operation)
        assert signature.parameters["save_to"].default is None
        assert signature.parameters["force"].default is False


def test_ssb_series_uses_inclusive_human_frame_bounds() -> None:
    """Series selection should not expose Python's half-open range contract."""
    from quantem.gpu import SSB

    parameters = inspect.signature(SSB.reconstruct_series).parameters

    assert "first_frame" in parameters
    assert "last_frame" in parameters
    assert "frames" not in parameters
    assert "screening_directory" not in parameters
    assert "fixed_probe" not in parameters
    assert parameters["probe_reference_frame"].default is None


def test_every_public_ssb_entry_point_defaults_to_full_bright_field() -> None:
    """C1c: all scientist entry points use the complete automatic BF disk."""

    from quantem.gpu import SSB

    for entry_point in (SSB, SSB.open):
        signature = inspect.signature(entry_point)
        assert signature.parameters["bright_field_intensity_threshold"].default == 0.0


def test_ssb_auto_backend_never_falls_back_to_cpu(monkeypatch) -> None:
    """C1d: unavailable GPU execution fails instead of changing science."""

    import pytest

    from quantem.gpu.device import select as device
    from quantem.gpu.ssb import workflow

    monkeypatch.setattr(device, "_cuda_probe", lambda: (False, "no CUDA device"))
    monkeypatch.setattr(device, "_mps_probe", lambda: (False, "no Apple GPU"))

    with pytest.raises(RuntimeError, match="No QuantEM GPU backend is available"):
        workflow.resolve("auto")


def test_reconstruct_series_reuses_only_exact_results(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    """The series API should route reuse through exact operation validation."""
    import json

    import numpy as np

    from quantem.gpu import SSB, SSBResult
    from quantem.gpu.ssb.persistence import FIT_SCHEMA

    raw = tmp_path / "raw"
    screen = raw / "live" / "screen"
    raw.mkdir()
    screen.mkdir(parents=True)
    reference = "91mm_30mrad_151"
    aberrations = {"C10_nm": -88.0, "C12_nm": 53.0, "phi12_deg": math.degrees(-0.78)}
    for index in range(1, 4):
        (raw / f"91mm_30mrad_{150 + index}_master.h5").write_bytes(b"raw")
        dataset = screen / f"91mm_30mrad_{150 + index}"
        dataset.mkdir()
        settings = {
            "voltage_kV": 80.0,
            "semiangle_mrad": 30.0,
            "scan_sampling_A": (0.2, 0.2),
            "scan_detector_rotation_deg": 173.0,
            "bright_field_radius_px": 55.0,
            "schema": FIT_SCHEMA,
            "aberrations": aberrations,
            "loss": 0.01 * index,
        }
        # a record as quantem.live screens it: the series stores a fit only in a record of this schema
        (dataset / "config.json").write_text(
            json.dumps({"schema": "quantem.screen-acquisition/2", "computed": {"ssb": settings}})
        )
        np.save(dataset / "bf.npy", np.full((8, 9), 20 + index))
        np.save(dataset / "df.npy", np.full((8, 9), 30 + index))

    class FakeSession:
        detector_sampling_mrad = (0.5, 0.5)

        def __init__(self, source):
            self.index = int(Path(source).stem.rsplit("_", 2)[-2]) - 150

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _result(self, phase):
            return SSBResult(
                object_wave=np.exp(
                    1j * np.full((8, 9), phase, dtype=np.float32)
                ).astype(np.complex64),
                backend="cuda",
                aberrations=aberrations,
                scan_detector_rotation_deg=173.0,
                loss=0.01 * self.index,
                reused=True,
            )

        def find_aberrations(self, **kwargs):
            assert kwargs["save_to"].name == "ssb-fit"
            return self._result(0.1 * self.index)

        def reconstruct(self, **kwargs):
            assert kwargs["aberrations"] == aberrations
            assert kwargs["save_to"].name == "ssb-locked"
            return self._result(0.3 + 0.1 * self.index)

    monkeypatch.setattr(
        SSB,
        "open",
        classmethod(lambda cls, source, **kwargs: FakeSession(source)),
    )

    independent = SSB.reconstruct_series(
        raw,
        first_frame=151,
        last_frame=153,
        progress=False,
    )
    fixed_probe_result = SSB.reconstruct_series(
        raw,
        first_frame=151,
        last_frame=153,
        probe_reference_frame=151,
    )
    assert "Checking exact saved SSB results" in capsys.readouterr().out

    np.testing.assert_allclose(independent.phase[:, 0, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(fixed_probe_result.phase[:, 0, 0], [0.4, 0.5, 0.6])
    np.testing.assert_array_equal(
        fixed_probe_result.bright_field[:, 0, 0],
        [21, 22, 23],
    )
    np.testing.assert_array_equal(
        fixed_probe_result.dark_field[:, 0, 0],
        [31, 32, 33],
    )
    assert fixed_probe_result.frames == (151, 152, 153)
    assert fixed_probe_result.datasets == (
        "91mm_30mrad_151",
        "91mm_30mrad_152",
        "91mm_30mrad_153",
    )
    assert not hasattr(fixed_probe_result, "fixed_probe")
    assert fixed_probe_result.probe_reference_frame == 151
    assert fixed_probe_result.probe_reference_dataset == reference
    assert [row["result"] for row in independent.records] == [
        "saved",
        "saved",
        "saved",
    ]
    assert [row["result"] for row in fixed_probe_result.records] == [
        "saved",
        "saved",
        "saved",
    ]
    independent_metadata = independent.metadata()["Value"]
    assert independent_metadata["Probe mode"] == "independent fits"
    assert "Probe reference frame" not in independent_metadata
    assert "Probe reference dataset" not in independent_metadata
    metadata = fixed_probe_result.metadata()["Value"]
    assert metadata["Requested backend"] == "auto"
    assert metadata["First frame"] == 151
    assert metadata["Last frame"] == 153
    assert metadata["Frame count"] == 3
    assert metadata["First dataset"] == "91mm_30mrad_151"
    assert metadata["Last dataset"] == "91mm_30mrad_153"
    assert metadata["Probe mode"] == "fixed probe"
    assert metadata["Probe reference frame"] == 151
    assert metadata["Probe reference dataset"] == reference
    assert metadata["Trials"] == 200
    assert metadata["Refinement"] == "nelder-mead"
    assert metadata["Results reused"] == 3


def test_series_writes_to_save_dir_and_forwards_flip_radius_and_unknown_rotation(tmp_path, monkeypatch) -> None:
    """``save_dir`` names the results folder; the flip, the fitted (unrounded) radius and a missing rotation reach SSB.open.

    The series always read and created ``<source>/live/screen`` (missing a Live output elsewhere), never forwarded a
    flip, rounded a saved 53.4 px radius to 53, and refused an acquisition without a recorded rotation.
    """
    import numpy as np

    from quantem.gpu import SSB, SSBResult, screening
    from quantem.gpu.ssb.persistence import FIT_SCHEMA

    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out" / "live" / "screen"
    for frame in (1, 2):
        (raw / f"scan_{frame}_master.h5").write_bytes(b"raw")
    saved = out / "scan_1"
    saved.mkdir(parents=True)
    (saved / "config.json").write_text(json.dumps({"schema": "quantem.screen-acquisition/2", "computed": {"ssb": {
        "schema": FIT_SCHEMA, "voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": [0.264, 0.264],
        "scan_detector_rotation_deg": 159.0, "scan_detector_rotation_flipped": True, "bright_field_radius_px": 53.4}}}))
    opened = []

    class FakeSession:
        detector_sampling_mrad = (0.55, 0.55)

        def __init__(self, kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def find_aberrations(self, **kwargs):
            rotation = self.kwargs["scan_detector_rotation_deg"]
            return SSBResult(object_wave=np.ones((4, 4), np.complex64), backend="mps",
                             aberrations={"C10_nm": 15.8, "C12_nm": 6.7, "phi12_deg": 17.5},
                             scan_detector_rotation_deg=158.9 if rotation is None else rotation,
                             scan_detector_rotation_flipped=False if rotation is None else self.kwargs["scan_detector_rotation_flipped"],
                             loss=0.1)

    def fake_open(cls, source, **kwargs):
        opened.append((Path(source).name, kwargs))
        return FakeSession(kwargs)

    prepared = []
    monkeypatch.setattr(SSB, "open", classmethod(fake_open))
    monkeypatch.setattr(screening, "prepare", lambda source, **kwargs: prepared.append(kwargs) or type("P", (), {
        name: np.zeros((4, 4), np.float32) for name in ("mean_diffraction_pattern", "bright_field", "dark_field",
                                                       "com_row", "com_col", "dpc_phase")})())
    (raw / "dataset.yaml").write_text("schema: quantem.dataset/2\nmicroscope: {voltage_kV: 300, semiangle_mrad: 30}\n"
                                      "calibrations: {default: {scan_sampling_A: [0.264, 0.264]}}\n")
    result = SSB.reconstruct_series(raw, first_frame=1, last_frame=2, save_dir=out, progress=False)

    assert result.results_directory == out.resolve() and not (raw / "live").exists()
    first, second = (kwargs for _, kwargs in opened)
    assert (first["scan_detector_rotation_deg"], first["scan_detector_rotation_flipped"], first["bright_field_radius_px"]) == (159.0, True, 53.4)
    assert second["scan_detector_rotation_deg"] is None and second["scan_detector_rotation_flipped"] is None
    assert [(p["scan_detector_rotation_deg"], p["scan_detector_rotation_flipped"]) for p in prepared] == [(159.0, True), (158.9, False)]
    assert (out / "scan_2" / "config.json").is_file()


def test_reconstruct_series_rejects_reversed_frame_bounds(tmp_path) -> None:
    """Inclusive frame bounds should fail clearly when reversed."""
    import pytest

    from quantem.gpu import SSB

    raw = tmp_path / "raw"
    screen = raw / "live" / "screen"
    raw.mkdir()
    screen.mkdir(parents=True)

    with pytest.raises(ValueError, match="first_frame must be at most last_frame"):
        SSB.reconstruct_series(
            raw,
            first_frame=3,
            last_frame=1,
            progress=False,
        )


def test_series_discovery_keeps_raw_acquisitions_after_partial_screening(
    tmp_path,
) -> None:
    """One saved screen result must not hide later raw acquisitions."""
    from quantem.gpu.ssb.series import screen_paths

    raw = tmp_path / "raw"
    raw.mkdir()
    for frame in (52, 53, 54):
        (raw / f"scan_{frame}_master.h5").touch()
    screened = raw / "live" / "screen" / "scan_52"
    screened.mkdir(parents=True)
    (screened / "config.json").write_text("{}", encoding="utf-8")

    paths = screen_paths(raw, raw / "live" / "screen")

    assert tuple(path.name for path in paths) == ("scan_52", "scan_53", "scan_54")


def test_series_bounds_are_acquisition_ids_not_list_positions(
    tmp_path,
    monkeypatch,
) -> None:
    """Non-consecutive endpoint IDs should include intervening acquisitions."""
    import numpy as np

    from quantem.gpu import SSB, SSBResult
    from quantem.gpu.ssb.persistence import FIT_SCHEMA

    raw = tmp_path / "raw"
    screen = raw / "live" / "screen"
    screen.mkdir(parents=True)
    for index, frame in enumerate((100, 250, 900), start=1):
        (raw / f"scan_{frame}_master.h5").write_bytes(b"raw")
        dataset = screen / f"scan_{frame}"
        dataset.mkdir()
        (dataset / "config.json").write_text(
            json.dumps(
                {
                    "schema": "quantem.screen-acquisition/2",
                    "computed": {
                        "ssb": {
                            "schema": FIT_SCHEMA,
                            "voltage_kV": 300.0,
                            "semiangle_mrad": 30.0,
                            "scan_sampling_A": (0.5, 0.5),
                            "scan_detector_rotation_deg": 0.0,
                            "aberrations": {
                                "C10_nm": 1.0,
                                "C12_nm": 2.0,
                                "phi12_deg": math.degrees(0.3),
                            },
                            "loss": 0.1,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        np.save(dataset / "ssb_phase.npy", np.full((2, 3), index))
        np.save(dataset / "bf.npy", np.zeros((2, 3), dtype=np.float32))
        np.save(dataset / "df.npy", np.zeros((2, 3), dtype=np.float32))

    class FakeSession:
        detector_sampling_mrad = (0.5, 0.5)

        def __init__(self, source):
            frame = int(Path(source).stem.rsplit("_", 2)[-2])
            self.index = {100: 1, 250: 2, 900: 3}[frame]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def find_aberrations(self, **kwargs):
            return SSBResult(
                object_wave=np.exp(
                    1j * np.full((2, 3), 0.1 * self.index, dtype=np.float32)
                ).astype(np.complex64),
                backend="cuda",
                aberrations={"C10_nm": 1.0, "C12_nm": 2.0, "phi12_deg": math.degrees(0.3)},
                scan_detector_rotation_deg=0.0,
                loss=0.1,
            )

    monkeypatch.setattr(
        SSB,
        "open",
        classmethod(lambda cls, source, **kwargs: FakeSession(source)),
    )

    result = SSB.reconstruct_series(
        raw,
        first_frame=100,
        last_frame=900,
        progress=False,
    )

    assert result.frames == (100, 250, 900)
    np.testing.assert_allclose(result.phase[:, 0, 0], (0.1, 0.2, 0.3))


def test_fixed_probe_series_fits_a_reference_stored_by_an_earlier_version(
    tmp_path,
    monkeypatch,
) -> None:
    """A probe reference whose stored fit lacks the current schema is fitted, never trusted (fit mocked at SSB.open)."""
    import numpy as np

    from quantem.gpu import SSB, SSBResult
    from quantem.gpu.ssb.persistence import FIT_SCHEMA

    raw = tmp_path / "raw"
    reference = raw / "live" / "screen" / "scan_1"
    reference.mkdir(parents=True)
    (raw / "dataset.yaml").write_text(
        "schema: quantem.dataset/2\n"
        "microscope: {voltage_kV: 300, semiangle_mrad: 30, scan_detector_rotation_deg: 12}\n"
        "calibrations: {mag: {scan_sampling_A: [0.5, 0.5]}}\n"
        "files: {1: {mag: mag}, 2: {mag: mag}}\n"
    )
    earlier = {"C10": 5.0, "C12": -0.3, "phi12": 0.18}
    # every value differs from dataset.yaml, so anything read from this block would show below; the record itself
    # is current (quantem.live's rewrite stamps the record and leaves an earlier fit block as it is)
    (reference / "config.json").write_text(json.dumps({"schema": "quantem.screen-acquisition/2", "computed": {"ssb": {
        "aberrations": earlier, "aberration_unit": "nm", "rotation_angle_deg": 0.0, "com_reversed": True, "loss": 0.014,
        "voltage_kV": 80.0, "semiangle_mrad": 20.0, "scan_sampling_A": (0.2, 0.2), "bright_field_radius_px": 55.0}}}))
    for frame in (1, 2):
        (raw / f"scan_{frame}_master.h5").write_bytes(b"raw")
        (raw / "live" / "screen" / f"scan_{frame}").mkdir(exist_ok=True)
        np.save(raw / "live" / "screen" / f"scan_{frame}" / "bf.npy", np.zeros((2, 3), dtype=np.float32))
        np.save(raw / "live" / "screen" / f"scan_{frame}" / "df.npy", np.zeros((2, 3), dtype=np.float32))
    fitted = {"C10_nm": -10.5, "C12_nm": 0.25, "phi12_deg": math.degrees(-0.5)}
    opened, fits, reconstructions = [], [], []

    class FakeSession:
        detector_sampling_mrad = (0.5, 0.5)

        def __init__(self, source):
            self.name = Path(source).name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _result(self, aberrations):
            return SSBResult(object_wave=np.ones((2, 3), dtype=np.complex64), backend="cuda", aberrations=aberrations,
                             scan_detector_rotation_deg=12.0, loss=0.25)

        def find_aberrations(self, **kwargs):
            fits.append(self.name)
            return self._result(fitted)

        def reconstruct(self, **kwargs):
            reconstructions.append((self.name, kwargs["aberrations"]))
            return self._result(kwargs["aberrations"])

    def open_session(cls, source, **kwargs):
        opened.append(kwargs)
        return FakeSession(source)

    monkeypatch.setattr(SSB, "open", classmethod(open_session))

    result = SSB.reconstruct_series(raw, first_frame=1, last_frame=2, probe_reference_frame=1, progress=False)

    assert fits == ["scan_1_master.h5"]
    assert reconstructions == [("scan_1_master.h5", fitted), ("scan_2_master.h5", fitted)]
    # every session opens with the settings of dataset.yaml: nothing comes from the earlier block
    assert opened == [{"backend": "auto", "voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": (0.5, 0.5),
                       "scan_detector_rotation_deg": 12.0, "scan_detector_rotation_flipped": False, "bright_field_radius_px": None}] * 3
    assert [record["C10 (nm)"] for record in result.records] == [-10.5, -10.5]
    stored = json.loads((reference / "config.json").read_text())["computed"]["ssb"]
    assert stored["schema"] == FIT_SCHEMA
    assert stored["aberrations"] == fitted
    assert (stored["scan_detector_rotation_deg"], stored["scan_detector_rotation_flipped"], stored["loss"]) == (12.0, False, 0.25)
    assert "com_reversed" not in stored and "rotation_angle_deg" not in stored


def test_series_takes_no_setting_from_a_fit_of_another_schema(tmp_path) -> None:
    """Without dataset.yaml or keyword settings an unmarked block supplies nothing, and the error names what is missing."""
    import pytest

    from quantem.gpu import SSB

    raw = tmp_path / "raw"
    folder = raw / "live" / "screen" / "scan_1"
    folder.mkdir(parents=True)
    (raw / "scan_1_master.h5").write_bytes(b"raw")
    (folder / "config.json").write_text(json.dumps({"computed": {"ssb": {
        "aberrations": {"C10": 5.0, "C12": -0.3, "phi12": 0.18}, "aberration_unit": "nm", "rotation_angle_deg": 0.0,
        "voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": (0.5, 0.5)}}}))
    before = (folder / "config.json").read_bytes()
    # the rotation is not missing: an acquisition without one has it estimated by SSB.open
    with pytest.raises(ValueError, match="missing voltage_kV, semiangle_mrad, scan_sampling_A\\. Run"):
        SSB.reconstruct_series(raw, first_frame=1, last_frame=1, progress=False)
    assert (folder / "config.json").read_bytes() == before


def test_series_reads_dataset_yaml_schema_2_only(tmp_path) -> None:
    """Each acquisition's settings from a hand-written quantem.dataset/2 file, in quantem.live's order.

    A per-file scan_detector_rotation_deg wins over the session's even when it is 0.0; the scan step is the file
    entry's own [row, col] pair, else its magnification's row, else calibrations.default. A file of another schema is
    refused with the rewrite named: its screen_rotation_deg and one-number step would give no rotation and no step.
    """
    import pytest

    from quantem.gpu.ssb.series import acquisition_settings, read_dataset_yaml

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dataset.yaml").write_text(
        """
schema: quantem.dataset/2
microscope: {voltage_kV: 300, semiangle_mrad: 30, scan_detector_rotation_deg: 12.0}
calibrations:
  default: {scan_sampling_A: [0.5, 0.6]}
  mag_5p1: {scan_sampling_A: [0.373, 0.41]}
files:
  1: {mag: mag_5p1, scan_detector_rotation_deg: 0.0}
  2: {mag: mag_5p1, scan_sampling_A: [0.2, 0.3]}
  3: {mag: mag_9p9, scan_detector_rotation_deg: 197.5}
""".strip()
    )
    document = read_dataset_yaml(raw)
    settings = {
        frame: acquisition_settings(raw / "live" / "screen" / f"scan_{frame}", document, voltage_kV=None,
                                    semiangle_mrad=None, scan_sampling_A=None, scan_detector_rotation_deg=None)
        for frame in (1, 2, 3, 4)
    }
    assert {frame: value["scan_detector_rotation_deg"] for frame, value in settings.items()} == {1: 0.0, 2: 12.0, 3: 197.5, 4: 12.0}
    assert {frame: value["scan_sampling_A"] for frame, value in settings.items()} == {
        1: (0.373, 0.41), 2: (0.2, 0.3), 3: (0.5, 0.6), 4: (0.5, 0.6)}
    assert all((value["voltage_kV"], value["semiangle_mrad"]) == (300, 30) for value in settings.values())

    (raw / "dataset.yaml").write_text(
        "schema_version: 1\nmicroscope: {voltage_kV: 300, semiangle_mrad: 30, screen_rotation_deg: 12}\n"
        "calibrations: {mag: {scan_sampling_A: 0.5}}\nfiles: {1: {mag: mag, rotation_deg: 0.0}}\n"
    )
    with pytest.raises(ValueError, match=r"schema None is not read.*quantem\.dataset/2.*tools/rewrite_dataset_yaml\.py"):
        read_dataset_yaml(raw)
    (raw / "dataset.yaml").write_text("schema: quantem.dataset/2\ncalibrations: {mag: {scan_sampling_A: 0.5}}\nfiles: {1: {mag: mag}}\n")
    with pytest.raises(ValueError, match=r"calibrations\.mag\.scan_sampling_A must be \[row, col\]"):
        acquisition_settings(raw / "live" / "screen" / "scan_1", read_dataset_yaml(raw), voltage_kV=None,
                             semiangle_mrad=None, scan_sampling_A=None, scan_detector_rotation_deg=None)


def test_series_matches_dataset_yaml_entries_by_master_name_and_range(tmp_path) -> None:
    """One resolver with quantem.live's matching (``formats.dataset_yaml.file_entry``).

    The series matched only the trailing number as an exact key, so a file under a range key ("06-24") or an entry
    keyed by name took the session default: 0.5 A here where quantem.live fits 0.264 and 0.373 A.
    """
    from quantem.gpu.formats import dataset_yaml
    from quantem.gpu.ssb.series import acquisition_settings, read_dataset_yaml

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dataset.yaml").write_text(
        """
schema: quantem.dataset/2
microscope: {voltage_kV: 300, semiangle_mrad: 30}
calibrations:
  mag_5p1: {scan_sampling_A: [0.373, 0.373]}
  mag_7p2: {scan_sampling_A: [0.264, 0.264]}
  default: {scan_sampling_A: [0.5, 0.5]}
files:
  "06-24": {mag: mag_7p2}
  first: {master: sample_01_master.h5, mag: mag_5p1}
  2: {master: zone_02_master.h5, mag: mag_5p1}
""".strip()
    )
    document = read_dataset_yaml(raw)
    steps = {
        name: acquisition_settings(raw / "live" / "screen" / name, document, voltage_kV=None, semiangle_mrad=None,
                                   scan_sampling_A=None, scan_detector_rotation_deg=0.0)["scan_sampling_A"]
        for name in ("sample_10", "sample_01", "sample_30", "sample_02")
    }
    # sample_02 shares number 2 with an entry naming another master: it takes the session default, not zone_02's step
    assert steps == {"sample_10": (0.264, 0.264), "sample_01": (0.373, 0.373), "sample_30": (0.5, 0.5), "sample_02": (0.5, 0.5)}
    assert dataset_yaml.acquisition_scan_sampling_A(document, "sample_10.qem") == ((0.264, 0.264), "dataset.yaml mag_7p2")
    scaled, source = dataset_yaml.acquisition_scan_sampling_A(document, "sample_40_master.h5", magnification_MX=3.6)
    assert scaled[0] == scaled[1] == 0.373 * 5.1 / 3.6 and source == "scaled from dataset.yaml mag_5p1 to 3.6 MX"


def test_reconstruct_series_initializes_missing_live_results(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """A raw series should run without a user-facing screening-directory step."""
    from types import SimpleNamespace

    import numpy as np

    from quantem.gpu import SSB, SSBResult
    from quantem.gpu import screening

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "scan_1_master.h5").write_bytes(b"raw")
    (raw / "dataset.yaml").write_text(
        """
schema: quantem.dataset/2
microscope:
  voltage_kV: 300
  semiangle_mrad: 30
  scan_detector_rotation_deg: 12
calibrations:
  mag: {scan_sampling_A: [0.5, 0.5]}
files:
  1: {mag: mag}
""".strip()
    )

    class FakeSession:
        detector_sampling_mrad = (0.5, 0.5)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def find_aberrations(self, **kwargs):
            return SSBResult(
                object_wave=np.full((3, 4), 1 + 1j, dtype=np.complex64),
                backend="cuda",
                aberrations={"C10_nm": 1.0, "C12_nm": 2.0, "phi12_deg": math.degrees(0.3)},
                scan_detector_rotation_deg=12.0,
                loss=0.25,
                bright_field_radius_px=8.0,
            )

    monkeypatch.setattr(
        SSB,
        "open",
        classmethod(lambda cls, source, **kwargs: FakeSession()),
    )
    zeros = np.zeros((3, 4), dtype=np.float32)
    monkeypatch.setattr(
        screening,
        "prepare",
        lambda *args, **kwargs: SimpleNamespace(
            mean_diffraction_pattern=np.zeros((5, 6), dtype=np.float32),
            bright_field=np.ones((3, 4), dtype=np.float32),
            dark_field=np.full((3, 4), 2.0, dtype=np.float32),
            com_row=zeros,
            com_col=zeros,
            dpc_phase=zeros,
        ),
    )

    result = SSB.reconstruct_series(
        raw,
        first_frame=1,
        last_frame=1,
    )
    assert (
        "Checking exact saved SSB results"
        in capsys.readouterr().out
    )

    assert result.phase.shape == (1, 3, 4)
    assert np.all(result.bright_field == 1.0)
    assert np.all(result.dark_field == 2.0)
    assert result.master_names == ("scan_1_master.h5",)
    assert result.probe_reference_frame is None
    assert result.records[0]["result"] == "computed"
    config = raw / "live" / "screen" / "scan_1" / "config.json"
    assert config.is_file()
    assert (config.parent / "bf.npy").is_file()
    assert (config.parent / "df.npy").is_file()


def test_mps_source_companion_precedes_historical_calibration(tmp_path) -> None:
    """C4: source-local exact columns, expect authoritative source ordering."""

    from quantem.gpu.ssb.mps.frames import bf_column_locations

    export = tmp_path / "current"
    source = export / "source" / "scan_master.h5"
    source.parent.mkdir(parents=True)
    source.touch()
    historical = tmp_path / "historical" / "calibration.json"

    candidates = bf_column_locations(str(source), str(historical))

    assert candidates == (export, source.parent, historical)


def test_mps_fit_calibration_restores_exact_export_brightfield(tmp_path) -> None:
    """C4: fit-only calibration, expect sibling exact BF geometry first."""

    from quantem.gpu.ssb.mps.frames import bf_column_locations

    source = tmp_path / "raw" / "scan_master.h5"
    source.parent.mkdir()
    source.touch()
    fit_calibration = tmp_path / "result" / "calibration.json"
    exact_calibration = fit_calibration.parent / "snapshots" / "cal.json"
    exact_calibration.parent.mkdir(parents=True)
    exact_calibration.write_text("{}", encoding="utf-8")

    candidates = bf_column_locations(str(source), str(fit_calibration))

    assert candidates == (
        exact_calibration,
        source.parent,
        fit_calibration,
    )


def test_open_screen_record_opens_at_every_stored_calibration(tmp_path, monkeypatch) -> None:
    """A Live screening record reopens with its fit's aberrations, rotation and flip, disk and both samplings.

    ``series.open_settings`` passed only the beam, step, rotation and radius, so the session derived its own detector
    sampling and disk centre and dropped the aberrations.
    """
    import pytest

    from quantem.gpu import SSB
    from quantem.gpu.ssb.persistence import FIT_SCHEMA
    from quantem.gpu.ssb.series import open_screen_record, screen_record_settings

    source = tmp_path / "scan_018_master.h5"
    source.write_bytes(b"raw")
    record = tmp_path / "out" / "live" / "screen" / "scan_018"
    record.mkdir(parents=True)
    fit = {"schema": FIT_SCHEMA, "voltage_kV": 300.0, "semiangle_mrad": 30.0, "scan_sampling_A": [0.264, 0.264],
           "detector_sampling_mrad": [0.552, 0.552], "aberrations": {"C10_nm": 15.77, "C12_nm": 6.7, "phi12_deg": 17.5},
           "scan_detector_rotation_deg": 339.78, "scan_detector_rotation_flipped": True,
           "bright_field_center_px": [95.5, 96.25], "bright_field_radius_px": 53.4}
    (record / "config.json").write_text(json.dumps({"schema": "quantem.screen-acquisition/2", "source_master": str(source),
                                                     "computed": {"ssb": fit}}))
    opened = []
    monkeypatch.setattr(SSB, "open", classmethod(lambda cls, path, **kwargs: opened.append((path, kwargs)) or "session"))

    assert open_screen_record(record) == "session"
    path, kwargs = opened[0]
    assert path == str(source)
    assert kwargs == {"backend": "auto", "verbose": False, **screen_record_settings(record)}
    assert (kwargs["scan_detector_rotation_deg"], kwargs["scan_detector_rotation_flipped"]) == (339.78, True)
    assert (kwargs["detector_sampling_mrad"], kwargs["bright_field_center_px"], kwargs["bright_field_radius_px"]) == (
        (0.552, 0.552), (95.5, 96.25), 53.4)
    assert kwargs["aberrations"]["C10_nm"] == 15.77
    (record / "config.json").write_text(json.dumps({"schema": "quantem.screen-acquisition/2", "computed": {"ssb": {}}}))
    with pytest.raises(ValueError, match="holds no SSB fit"):
        open_screen_record(record)
