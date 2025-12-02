# Chatterbox TTS - Flask UI
<img width="1193" height="742" alt="Screenshot 2025-12-02 at 22 56 59" src="https://github.com/user-attachments/assets/fe5e57ac-5827-4429-b988-98cdd039189c" />

# Chatterbox Multilingual TTS – Flask UI

Minimal Flask web UI for [ResembleAI Chatterbox-TTS](https://github.com/resemble-ai/chatterbox), with:

# Chatterbox Multilingual TTS – Flask UI

Minimal Flask web UI for [ResembleAI Chatterbox-TTS](https://github.com/resemble-ai/chatterbox), with:

- 23-language multilingual TTS (ChatterboxMultilingualTTS)
- Zero-shot voice cloning via reference audio
- Controls for **exaggeration**, **temperature**, **CFG / pace**, and **seed**
- Simple HTML/JS front-end (no Gradio)

> ⚠️ First run will download ~3–4 GB of model weights from Hugging Face.  
> Make sure there is enough disk space and a stable connection.

---

## 1. Requirements

Tested on:

- macOS (Apple Silicon, M-series)
- Python **3.11**
- `git`

GPU is **not required** – it runs on CPU, just slower.

---

## 2. Clone the repo

```bash
git clone https://github.com/sylwesterdigital/chatterbox-flask.git
cd chatterbox-flask
```

---

## 3. Create and activate virtualenv

Use Python 3.11 (adjust path if needed):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` in your shell prompt.

Upgrade `pip` inside this venv:

```bash
python -m pip install --upgrade pip
```

---

## 4. Install dependencies

### 4.1 Core runtime

```bash
python -m pip install \
  "numpy>=1.24.0,<1.26.0" \
  flask \
  "torch==2.6.0" \
  "torchaudio==2.6.0"
```

> On Apple Silicon, the wheel selection happens automatically when using Python 3.11.

### 4.2 Install chatterbox (local, editable, *no deps*)

This repo already contains the Chatterbox code under `src/`, so install it in editable mode:

```bash
python -m pip install -e . --no-deps
```

If your Python still complains about an “externally managed environment”, append:

```bash
python -m pip install -e . --no-deps --break-system-packages
```

(Inside a venv this is safe; it only affects this virtualenv.)

### 4.3 Extra libs used by the Flask app

```bash
python -m pip install \
  "librosa==0.11.0" \
  "transformers==4.46.3" \
  "diffusers==0.29.0" \
  "resemble-perth==1.0.1" \
  "conformer==0.3.2" \
  "safetensors==0.5.3" \
  "s3tokenizer==0.2.0"
```

> Chinese segmentation (`pkuseg`) is optional – if it fails to build, the model still works; Chinese text just uses a simpler tokenizer and will log a warning.

---

## 5. (Optional) Hugging Face token

By default the model downloads from:

* `ResembleAI/chatterbox` on Hugging Face

If you need a private token, set:

```bash
export HF_TOKEN=your_hf_token_here
```

before running the app.

---

## 6. Run the Flask app

From the project root with the venv active:

```bash
source .venv/bin/activate   # if not already active
python app.py
```

You should see something like:

```text
* Serving Flask app 'app'
* Debug mode: on
* Running on http://127.0.0.1:8000
```

Open in a browser:

* [http://127.0.0.1:8000](http://127.0.0.1:8000)

On **first request**, the app will:

* Download Chatterbox multilingual weights (~3–4 GB)
* Cache them under your Hugging Face cache directory
* Load the model onto CPU

Subsequent runs will reuse the cached weights and start much faster.

---

## 7. Using the UI

1. **Text to synthesize**
   Type up to ~300 characters of text.

2. **Language**
   Choose any of the 23 supported languages from the dropdown.

3. **Reference voice (optional)**

   * Upload an audio file (e.g. `.wav`, `.m4a`, `.mp3`) as a voice profile, **or**
   * Tick “Use built-in sample voice” to use the default demo reference for that language.

4. **Controls**

   * **Exaggeration**
     Expressiveness.

     * ≈0.5 → neutral
     * Higher → more emotional / dramatic.
   * **CFG / Pace**
     Guidance + pacing. Lower values can slow down speech and reduce over-guidance.
   * **Temperature**
     Sampling randomness. Higher = more varied output.
   * **Seed**
     `0` = random each time; any other integer makes output more deterministic.

5. Click **“Generate Audio”**

   * The right panel shows the last output and lets you replay it.
   * You can tweak parameters and generate again.

---

## 8. Troubleshooting

* **`ModuleNotFoundError: No module named 'numpy'`**
  The venv is either not active or did not get dependencies installed.

  * Check prompt shows `(.venv)`
  * Re-run the install steps in sections 3–4.

* **PEP 668 / “externally-managed-environment”**
  Make sure you are inside the **local `.venv`**, then use:

  ```bash
  python -m pip install --break-system-packages -e . --no-deps
  ```

* **Very slow or stuck on first generation**
  The model weights are large and may take several minutes on the very first download + load. Later generations will be much faster.

---

## 9. License

This repo is based on ResembleAI’s Chatterbox-TTS and follows the original MIT license (see `LICENSE`).

