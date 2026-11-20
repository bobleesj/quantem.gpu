"""A streamed session's index and session log, joined by time."""

from datetime import datetime, timezone
import json
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
    (tmp_path / "index.jsonl").write_text("".join(checked(row) + "\n" for row in scans) + '{"session": "nig')
    events = [
        {"session": "night", "event": "reading", "time_utc": utc(T0 - 5), "changed": [], "defocus_nm": -10.0,
         "beam_tilt_mrad": [0.0, 0.0]},
        {"session": "night", "event": "scan_startup", "time_utc": utc(T0), "series_id": 7, "beam_on_epoch_s": T0},
        {"session": "night", "event": "reading", "time_utc": utc(T0 + 4), "changed": ["defocus_nm"],
         "defocus_nm": -30.0, "beam_tilt_mrad": [0.0, 0.0]},
        {"session": "other", "event": "reading", "time_utc": utc(T0 + 1), "changed": ["defocus_nm"], "defocus_nm": 99.0},
    ]
    (tmp_path / "session-log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    return tmp_path


def test_scans_times_and_the_clock_pair(folder):
    session = io.session(folder, session="night")
    assert session.count == 2000 and len(session.scans) == 2
    assert session.clock_offset_s == pytest.approx(OFFSET) and session.clock_spread_s == 0
    assert session.time(0) == pytest.approx(T0)
    assert session.time(1500) == pytest.approx(T0 + 100 + 5.0)       # scan 2 has no start of its own: the offset places it
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
