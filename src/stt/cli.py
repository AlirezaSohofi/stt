from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import typer

from stt.config import SttConfig, load_config
from stt.devices import print_input_devices_report, run_input_meter
from stt.doctor import run_doctor
from stt.models import list_models, validate_model
from stt.repl import run_once, run_repl
from stt.transcriber import WhisperTranscriber
from stt.util import eprint

app = typer.Typer(add_completion=False, invoke_without_command=True, no_args_is_help=False)


def _apply_cli_model(cfg: SttConfig, model: Optional[str]) -> None:
    """Override cfg.model from a subcommand-specific --model / -m (validated)."""
    if model is None or not str(model).strip():
        return
    name = str(model).strip()
    try:
        validate_model(name)
    except ValueError as e:
        eprint(f"error: {e}")
        raise typer.Exit(2) from e
    cfg.model = name


@app.callback()
def _cli_callback(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to config.toml",
        envvar="STT_CONFIG",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model (before subcommand, e.g. stt --model base once); see also each command's --model",
        envvar="STT_MODEL",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose status on stderr"),
    debug: bool = typer.Option(False, "--debug", help="Debug diagnostics on stderr"),
) -> None:
    """Local speech-to-text (Whisper + microphone)."""
    overrides: dict[str, Any] = {}
    if model is not None and model.strip():
        overrides["model"] = model.strip()
    sub = ctx.invoked_subcommand
    strict_model = sub not in ("doctor", "devices", "meter")
    try:
        cfg = load_config(
            config,
            cli_overrides=overrides or None,
            strict_model=strict_model,
        )
    except (ValueError, OSError) as e:
        eprint(f"error: {e}")
        raise typer.Exit(1) from e

    ctx.obj = {
        "cfg": cfg,
        "config_path": config,
        "verbose": verbose,
        "debug": debug,
    }

    if ctx.invoked_subcommand is None:
        code = run_repl(cfg, verbose=verbose, debug=debug)
        raise typer.Exit(code)


@app.command("repl")
def cmd_repl(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model name (tiny, base, small, …)",
        envvar="STT_MODEL",
    ),
) -> None:
    o = ctx.obj
    cfg = o["cfg"]
    _apply_cli_model(cfg, model)
    raise typer.Exit(run_repl(cfg, verbose=o["verbose"], debug=o["debug"]))


@app.command("once")
def cmd_once(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model name (tiny, base, small, …)",
        envvar="STT_MODEL",
    ),
) -> None:
    o = ctx.obj
    cfg = o["cfg"]
    _apply_cli_model(cfg, model)
    raise typer.Exit(run_once(cfg, verbose=o["verbose"], debug=o["debug"]))


@app.command("transcribe")
def cmd_transcribe(
    ctx: typer.Context,
    audio_path: Path = typer.Argument(..., exists=True, readable=True),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model name (tiny, base, small, …)",
        envvar="STT_MODEL",
    ),
) -> None:
    o = ctx.obj
    cfg = o["cfg"]
    _apply_cli_model(cfg, model)
    verbose: bool = o["verbose"]
    debug: bool = o["debug"]
    if not audio_path.is_file():
        eprint(f"error: not a file: {audio_path}")
        raise typer.Exit(2)
    transcriber = WhisperTranscriber(cfg)
    if verbose:
        eprint(f"transcribing with model={transcriber.model_name}")
    if debug:
        eprint(f"debug: path={audio_path}")
    try:
        text = transcriber.transcribe_file(audio_path)
    except Exception as e:
        eprint(f"error: failed to transcribe audio: {e}")
        raise typer.Exit(1) from e
    print(text, flush=True)


@app.command("models")
def cmd_models() -> None:
    for name in list_models():
        print(name)


@app.command("doctor")
def cmd_doctor(ctx: typer.Context) -> None:
    o = ctx.obj
    cfg = o["cfg"]
    path = o.get("config_path")
    code = run_doctor(cfg, config_path=path)
    raise typer.Exit(code)


@app.command("devices")
def cmd_devices(ctx: typer.Context) -> None:
    print_input_devices_report(ctx.obj["cfg"].default_device)


@app.command("meter")
def cmd_meter(
    ctx: typer.Context,
    seconds: float = typer.Option(5.0, "--seconds", "-s", help="How long to listen"),
    device: Optional[int] = typer.Option(
        None,
        "--device",
        "-d",
        help="Input device index (default: config default_device or system default)",
    ),
) -> None:
    cfg = ctx.obj["cfg"]
    dev = device if device is not None else cfg.default_device
    code = run_input_meter(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        device=dev,
        seconds=seconds,
    )
    raise typer.Exit(code)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
