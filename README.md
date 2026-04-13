# stt

Local speech-to-text REPL for macOS: record from the microphone, transcribe with [OpenAI Whisper](https://github.com/openai/whisper), print text to stdout.


## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) on `PATH` (e.g. `sudo port install ffmpeg`)
- Microphone permission when prompted (macOS)

Optional: if installing Whisper fails building `tiktoken`, install a Rust toolchain or use prebuilt wheels.

## Install

```bash
uv sync --extra dev
```

## Usage

```bash
uv run stt              # interactive REPL (default)
uv run stt repl
uv run stt once         # one recording, then stop
uv run stt transcribe path/to/file.wav
uv run stt models
uv run stt doctor
uv run stt devices   # list input device indices (pick default_device)
uv run stt meter     # live peak meter for ~5s — use this if recordings are silent
```

**Whisper model** (for speed vs accuracy): use **`--model`** or **`-m`** on `repl`, `once`, and `transcribe`, or **before** the subcommand on the main app (e.g. `uv run stt --model base once`). Examples:

```bash
uv run stt once --model base
uv run stt once -m tiny | pbcopy  # copy text to the clipboard
uv run stt repl --model small.en
uv run stt transcribe -m base ./clip.wav
uv run stt --model tiny              # default REPL with tiny
```

`uv run stt models` lists valid names. Same value can be set via `STT_MODEL` or `model` in config.

In the REPL: **Enter** starts recording, **Enter** again (or `stop`) stops and transcribes. Exit with **`q`**, **`quit`**, **`exit`**, **`esc`**, or **`escape`** (any case). While recording, finish with Enter/`stop` first, then quit. With line-based input, type **`esc`** or **`q`**; the physical Esc key is not handled separately. **`stt once`**: **`q`** / **`esc`** / **`quit`** / **`exit`** cancels without transcribing. Status and interactive prompts (`>`, `recording>`) go to **stderr**; only the transcript is printed to **stdout**, so piping (e.g. `stt once | pbcopy`) stays clean.

### Where is the microphone? What is loopback?

On **MacBook Air / Pro**, the built-in mic is usually a tiny hole (or a cluster of holes) **next to the speaker grille**—often **near the keyboard** (left/right of the top row on many models) or along the **bottom edge of the display**. Exact placement varies by model year; check [Apple’s support pages](https://support.apple.com/) for your Mac.


### Capture quality (macOS)

Audio is recorded from the **selected microphone**, not from app or browser output. Playing TTS (e.g. Google Translate) through **speakers** often reaches the mic as a **very quiet** signal; Whisper may then guess short phrases that are wrong (e.g. “Thank you”). Prefer **speaking into the mic**, use a **headset**, turn the speaker volume up and reduce room noise, or use a **virtual loopback** device (e.g. BlackHole) as the input in System Settings and set `default_device` in config to that device’s index (inspect indices with `uv run python -c "import sounddevice as sd; print(sd.query_devices())"`).

After each recording, stderr includes a short **level summary** (duration, peak, RMS). If you see a **warning** about a quiet or empty signal, fix the input path before trusting the transcript.

Set `[transcription] verbose = true` in config only if you want Whisper’s detailed decode logging. The default (`false`) keeps Whisper quiet (no progress bar or “Detected language” line on stderr).

### If recordings are completely silent (peak = 0)

That almost always means **no audio is reaching PortAudio**, not that you are too quiet.

1. **Microphone permission for the app that runs `stt`**  
   macOS gates the mic per app. Open **System Settings → Privacy & Security → Microphone** and turn **on** the app you use to run commands (**Terminal**, **iTerm2**, **Cursor**, **VS Code**, **Warp**, etc.). If you started `stt` from Cursor’s integrated terminal, enable **Cursor**; if from Terminal.app, enable **Terminal**. Change the toggle, then **fully quit and reopen** that app.

2. **Correct input device and input level**  
   **System Settings → Sound → Input** — select **MacBook Microphone** (or your headset), and move **Input volume** so the meter moves when you talk. If the OS meter does not move, `stt` will not hear you either.

3. **Confirm what `stt` is using**  
   Run `uv run stt devices` and note which line is marked **(default input)**. If that is not your real mic (e.g. a virtual device), set `default_device = <index>` under the correct line in `~/.config/stt/config.toml`.

4. **Live test**  
   Run `uv run stt meter` and speak; you should see **peak** and the bar move. Try `uv run stt meter --device N` for other indices from `stt devices`. Exit code `1` after a silent meter run means the problem is still permission, device, or volume—not Whisper.

## Configuration

Optional file: `~/.config/stt/config.toml`. Precedence: CLI flags, environment variables, config file, defaults.

Environment variables include `STT_MODEL` (same as `--model` / `-m`), `STT_LANGUAGE`, `STT_SAVE_RECORDINGS`, `STT_SESSIONS_DIR`, `STT_CONFIG`, `STT_DEFAULT_DEVICE`, `STT_SAMPLE_RATE`, `STT_CHANNELS`.

## Sessions

With `save on` in the REPL or `save_recordings = true` in config, each utterance is stored under `sessions_dir` (default `~/.local/share/stt/sessions/<timestamp>/`) as `audio.wav`, `transcript.txt`, and `metadata.json`.

## Future

The `last` command (re-print previous transcript) is not implemented in v1.
