"""Keep generated arrays, loading indexes and subprocess temp files test-owned."""

import os
import signal
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def disposable_loading_files(tmp_path_factory: pytest.TempPathFactory):
    """Remove temporary loading files even when a test fails or is interrupted.

    Explicit output paths remain the caller's responsibility. Tests must write
    numerical exports to ``tmp_path``, never beside original acquisitions.

    Examples
    --------
    A test calling ``tempfile.mkdtemp()`` or launching a native loader inherits
    this session's disposable directory without an additional fixture argument.
    """
    previous_tempdir = tempfile.tempdir
    previous_handler = signal.getsignal(signal.SIGTERM)
    variables = ("TMPDIR", "TMP", "TEMP")
    previous_environment = {name: os.environ.get(name) for name in variables}

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("Test cancelled; cleaning generated loading files")

    with tempfile.TemporaryDirectory(
        prefix="loading-", dir=tmp_path_factory.getbasetemp()
    ) as directory:
        try:
            tempfile.tempdir = directory
            for name in variables:
                os.environ[name] = directory
            signal.signal(signal.SIGTERM, interrupt)
            yield
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            tempfile.tempdir = previous_tempdir
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
