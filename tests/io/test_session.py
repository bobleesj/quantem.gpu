"""A streamed session's index and session log, joined by time."""

from datetime import datetime, timezone
from importlib import import_module
import json
from types import SimpleNamespace
import zlib

import pytest

from quantem.gpu import io


def checked(record):
    body = json.dumps(record, allow_nan=False, sort_keys=True)
    return json.dumps({**record, "crc32": zlib.crc32(body.encode())})


def utc(epoch_s):
    return datetime.fromtimestamp(epoch_s, timezone.utc).isoformat()


T0 = 1_800_000_000.0        # the microscope PC's clock when scan 1's beam went on
OFFSET = 0.25               # the receiver's clock is this far ahead


@pytest.fixture
def folder(tmp_path):
    scans = [
        {"session": "night", "series_id": 7, "series": "series_7", "first_index": 0, "expected": 1000,
         "received": 1000, "begin_epoch_s": T0 + OFFSET, "scan_pattern": "raster", "dwell_s": 0.01},
        {"session": "night", "live_frames": 4, "begin_epoch_s": T0 + 20, "end_epoch_s": T0 + 25},
        {"session": "night", "series_id": 8, "series": "series_8", "first_index": 1000, "expected": 1000,
         "received": 1000, "begin_epoch_s": T0 + 100 + OFFSET, "scan_pattern": "raster", "dwell_s": 0.01},
        {"session": "other", "series_id": 1, "series": "series_1", "first_index": 0, "expected": 10,
         "received": 10, "begin_epoch_s": T0, "scan_pattern": "raster", "dwell_s": 0.01},
    ]
    complete = "".join(checked(row) + "\n" for row in scans).encode()
    (tmp_path / "index.jsonl").write_bytes(complete + b'{"session": "nig\xc3')
    events = [
        {"session": "night", "event": "reading", "time_utc": utc(T0 - 5), "changed": [], "defocus_nm": -10.0,
         "beam_tilt_mrad": [0.0, 0.0]},
        {"session": "night", "event": "scan_startup", "time_utc": utc(T0), "series_id": 7, "beam_on_epoch_s": T0},
        {"session": "night", "event": "reading", "time_utc": utc(T0 + 4), "changed": ["defocus_nm"],
         "defocus_nm": -30.0, "beam_tilt_mrad": [0.0, 0.0]},
        {"session": "other", "event": "reading", "time_utc": utc(T0 + 1), "changed": ["defocus_nm"], "defocus_nm": 99.0},
        {"session": "night", "event": "reading", "time_utc": utc(T0 + 50), "changed": ["scan_rotation_deg", "beam_tilt_mrad"],
         "defocus_nm": -30.0, "scan_rotation_deg": 90.0, "beam_tilt_mrad": [2.0, 0.0]},
        {"session": "night", "event": "scan_beam_on", "time_utc": utc(T0 + 100.5), "series_id": 8,
         "beam_on_epoch_s": T0 + 100.5, "pair_rotation_deg": 90, "scan_rotation_deg": 91.5},
        {"session": "night", "event": "reading", "time_utc": utc(T0 + 103), "changed": ["scan_rotation_deg"],
         "defocus_nm": -30.0, "scan_rotation_deg": 1.5, "beam_tilt_mrad": [2.0, 0.0]},      # written late, after the restore
    ]
    (tmp_path / "session-log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    return tmp_path


def test_scans_times_and_the_clock_pair(folder):
    session = io.session(folder, session="night")
    assert session.count == 2000 and len(session.scans) == 2
    assert session.clock_offset_s == pytest.approx(0.0, abs=1e-6)
    assert session.time(0) == pytest.approx(T0)
    assert session.time(1500) == pytest.approx(T0 + 100.5 + 5.0)     # every scan's own beam-on line places it
    assert [row["clock"] for row in session.index] == ["logged", "logged"]
    assert session.clock_spread_s == pytest.approx(0.5)              # the two scans' pairs disagree by this much
    assert "2 scans" in repr(session) and "0 of 2 files" in repr(session)


def test_a_reading_is_reported_with_its_age_and_a_change_as_an_interval(folder):
    session = io.session(folder, session="night")
    early = session.state(100)["defocus_nm"]                         # 1 s into scan 1
    assert early == {"value": -10.0, "age_s": pytest.approx(6.0), "between": None, "indices": None}
    inside = session.state(390)["defocus_nm"]                        # 3.9 s: within one read of the change at 4 s
    assert inside["between"] == (-10.0, -30.0)
    assert inside["indices"][0] in (379, 380) and inside["indices"][1] == 400     # 3.8 s to 4.0 s, to the clock's rounding
    late = session.state(600)["defocus_nm"]
    assert late["value"] == -30.0 and late["between"] is None and late["age_s"] == pytest.approx(2.0)
    assert session.state(600)["beam_tilt_mrad"]["value"] == [0.0, 0.0]


def test_a_folder_of_several_sessions_asks_which(folder):
    with pytest.raises(ValueError, match="choose one with session="):
        io.session(folder)
    with pytest.raises(ValueError, match="No session 'day'"):
        io.session(folder, session="day")


def test_a_scan_that_is_not_in_the_folder_says_so(folder):
    session = io.session(folder, session="night")
    with pytest.raises(FileNotFoundError, match="series_8"):
        session.scans[1]
    with pytest.raises(IndexError):
        session.state(2000)


def test_each_scan_of_a_go_gets_its_own_state_from_the_log(folder):
    table = io.session(folder, session="night").report()
    assert list(table["first_index"]) == [0, 1000]
    assert list(table["defocus_nm"]) == [-10.0, -30.0]               # read once at Go in the header; here per scan
    assert list(table["beam_tilt_mrad"]) == [[0.0, 0.0], [2.0, 0.0]]
    assert table["scan_rotation_deg"][1] == 91.5 and table["file"][1] is None     # set for scan 2, in its own line
    session = io.session(folder, session="night")
    first = session.state(1000)["scan_rotation_deg"]
    assert first == {"value": 91.5, "age_s": pytest.approx(0.0, abs=1e-6), "between": None, "indices": None}
    assert session.state(1000)["pair_rotation_deg"]["value"] == 90


def test_a_value_set_for_one_scan_alone_does_not_follow_into_the_next_scan(tmp_path):
    scans = [{"session": "pair", "series_id": n, "series": f"series_{n}", "first_index": (n - 1) * 100, "expected": 100,
              "received": 100, "begin_epoch_s": T0 + 10 * n, "scan_pattern": "raster", "dwell_s": 0.01} for n in (1, 2, 3)]
    (tmp_path / "index.jsonl").write_text("".join(checked(row) + "\n" for row in scans))
    days = [tmp_path / "logs" / "2026-11-19", tmp_path / "logs" / "2026-11-20"]
    for day in days:
        day.mkdir(parents=True)
    first = [{"session": "pair", "event": "reading", "time_utc": utc(T0), "changed": [], "scan_rotation_deg": 0.0},
             {"session": "pair", "event": "scan_beam_on", "time_utc": utc(T0 + 10), "series_id": 1, "beam_on_epoch_s": T0 + 10,
              "pair_rotation_deg": 0, "scan_rotation_deg": 0.0},
             {"session": "pair", "event": "scan_beam_on", "time_utc": utc(T0 + 20), "series_id": 2, "beam_on_epoch_s": T0 + 20,
              "pair_rotation_deg": 90, "scan_rotation_deg": 90.0}]
    second = [{"session": "pair", "event": "scan_beam_on", "time_utc": utc(T0 + 30), "series_id": 3, "beam_on_epoch_s": T0 + 30}]
    (days[0] / "session-log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in first))
    (days[1] / "session-log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in second))
    session = io.session(tmp_path, log=tmp_path / "logs")                    # the folder of days: both are read
    table = session.report(names=("scan_rotation_deg", "pair_rotation_deg"))
    assert list(table["clock"]) == ["logged", "logged", "logged"]
    assert list(table["scan_rotation_deg"]) == [0.0, 90.0, 0.0]              # the plain scan after the pair: the last reading
    assert table["pair_rotation_deg"][1] == 90 and table["pair_rotation_deg"].isna()[2]   # nothing was set for scan 3


def test_qem_without_index_uses_the_shape_from_its_file_header(tmp_path, monkeypatch):
    path = tmp_path / "scan.qem"
    path.touch()
    header = SimpleNamespace(
        scan_shape=(2, 3),
        metadata={"session": "night", "series_id": 7, "first_index": 12, "series": "series_7"},
    )

    def read_header(source):
        assert source == path
        return header

    monkeypatch.setattr(import_module("quantem.gpu.io.inspect"), "inspect", read_header)
    saved = io.session(tmp_path)

    assert saved.count == 18
    assert saved.index[0]["received"] == 6
    assert saved.index[0]["path"] == path


def test_reused_detector_series_id_keeps_the_clock_for_each_scan(tmp_path):
    scans = [
        {"session": "night", "series_id": 7, "first_index": first, "received": 10,
         "begin_epoch_s": T0 + delay + OFFSET, "dwell_s": 0.01}
        for first, delay in ((0, 0), (10, 100))
    ]
    events = [
        {"session": "night", "event": kind, "time_utc": utc(T0 + delay),
         "series_id": 7, "beam_on_epoch_s": T0 + delay}
        for kind, delay in (("scan_startup", 0), ("scan_beam_on", 0), ("scan_beam_on", 100))
    ]
    (tmp_path / "index.jsonl").write_text("".join(checked(row) + "\n" for row in scans))
    log = tmp_path / "session-log.jsonl"
    log.write_text("".join(json.dumps(event) + "\n" for event in events))

    saved = io.session(tmp_path)

    assert saved.time(0) == pytest.approx(T0, abs=1e-5)
    assert saved.time(10) == pytest.approx(T0 + 100, abs=1e-5)
    assert saved.clock_offset_s == pytest.approx(OFFSET)
