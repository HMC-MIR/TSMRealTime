## TSM RealTime

GUI app to compare tempo‑modification (TSM) audio algorithms in real time.

### Features
- **Real‑time GUI**: Tkinter interface to load audio and switch algorithms
- **Audio stack**: NumPy/SciPy/Librosa, optional PyAudio for capture/playback
- **Cross‑platform**: macOS, Windows, Linux (where PortAudio is available)

## Installation

### Requirements
- **Python**: 3.8+ (the project currently targets modern 3.x)
- **FFmpeg**: recommended for broader format support
- **PortAudio** (for PyAudio): required if you use microphone/streaming
  - macOS: `brew install portaudio`
  - Ubuntu/Debian: `sudo apt-get install portaudio19-dev`
  - Windows: PyAudio wheels are often available; if build fails, install a prebuilt wheel.

### From PyPI
```bash
pip install tsm-realtime
```

Run the app:
```bash
tsm-realtime-gui
```

### From source (this repo)
```bash
git clone https://github.com/HMC-MIR/TSMRealTime.git
cd TSMRealTime
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -e .
```

Run the app:
```bash
tsm-realtime-gui
```

If you prefer a standard (non‑editable) install from a local build:
```bash
python -m build
pip install dist/*.whl
```

## Usage

### CLI entry point
After install, a console script `tsm-realtime-gui` is available. It launches the Tkinter GUI, which allows selecting audio files (e.g., from `app/samples/`) and comparing algorithms.

### Module entry point
You can also launch via Python:
```bash
python -c "import app; app.main()"
```

## Development

### Editable install
```bash
pip uninstall -y tsm-realtime  # optional cleanup
pip install -e .[dev]
```

Now edits in the `app/` package are picked up immediately.

### Lint/Test (add if you adopt tools)
- Lint: `ruff check .` or `flake8`
- Format: `ruff format .` or `black .`
- Tests: `pytest`

## Packaging and publishing

Ensure `pyproject.toml` has the correct metadata (name `tsm-realtime`, version bump, scripts). Then:
```bash
pip install --upgrade build twine
python -m build
python -m twine upload dist/*
```

Install from PyPI to verify:
```bash
pip install --no-cache-dir --force-reinstall tsm-realtime
tsm-realtime-gui
```

## Troubleshooting

- **Changed code not reflected when running**
  - If installed via wheel: bump version, rebuild, and reinstall; or use `pip install -e .` for editable mode.
  - Clean artifacts: `rm -rf build dist ./*.egg-info` then rebuild.

- **PyAudio fails to install (macOS/Linux)**
  - Install PortAudio headers first (see Requirements), then reinstall: `pip install --no-cache-dir pyaudio`.

- **Command not found: dsp-audio-gui**
  - Ensure your environment’s `bin`/`Scripts` directory is on PATH and the package installed successfully: `pip show tsm-realtime`.

- **Large package size**
  - Sample audio files may bloat wheels. If you don’t need them in the wheel, remove `"samples/*"` from `[tool.setuptools.package-data]` and rebuild.

## License

MIT. See `pyproject.toml` for metadata and the repository for details.

# TSM-RealTime