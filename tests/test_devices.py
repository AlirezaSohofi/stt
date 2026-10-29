from __future__ import annotations

from unittest.mock import MagicMock, patch

from stt.devices import _meter_bar, describe_device_line, print_input_devices_report


def test_meter_bar() -> None:
    assert "#" in _meter_bar(0.5)


def test_describe_device_line() -> None:
    line = describe_device_line(
        3,
        {"name": "Test Mic", "max_input_channels": 1, "hostapi": 0},
        mark_default=True,
    )
    assert "[3]" in line and "Test Mic" in line and "default input" in line


@patch("stt.devices.sd.query_hostapis")
@patch("stt.devices.sd.query_devices")
@patch("stt.devices.sd.default")
def test_print_input_devices_report(mock_default, mock_query_devices, mock_hostapis) -> None:
    mock_default.device = (1, 2)
    mock_query_devices.return_value = [
        {"name": "Out", "max_input_channels": 0, "hostapi": 0},
        {"name": "MacBook Mic", "max_input_channels": 1, "hostapi": 0},
    ]
    mock_hostapis.side_effect = lambda _i: {"name": "Core Audio"}
    print_input_devices_report(cfg_device=None)
    mock_query_devices.assert_called()


_DEVS = [
    {"name": "Out", "max_input_channels": 0, "hostapi": 0},
    {"name": "MacBook Pro Microphone", "max_input_channels": 1, "hostapi": 0},
    {"name": "AirPods Pro", "max_input_channels": 1, "hostapi": 0},
]


def test_parse_device_pref() -> None:
    from stt.devices import parse_device_pref

    assert parse_device_pref("default") is None
    assert parse_device_pref("") is None
    assert parse_device_pref("2") == 2
    assert parse_device_pref(" AirPods ") == "AirPods"


@patch("stt.devices.sd.query_devices", return_value=_DEVS)
def test_resolve_input_device(_mock) -> None:
    from stt.devices import resolve_input_device

    assert resolve_input_device(None) == (None, None)
    assert resolve_input_device("airpods") == (2, None)
    assert resolve_input_device(1) == (1, None)
    idx, warn = resolve_input_device(5)
    assert idx is None and "not available" in warn
    idx, warn = resolve_input_device(0)  # output-only device
    assert idx is None and warn
    idx, warn = resolve_input_device("Yeti")
    assert idx is None and "Yeti" in warn
