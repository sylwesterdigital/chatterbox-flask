Good. Torch/torchaudio are in. Do this next, **from `/Users/smielniczuk/Documents/works/chatterbox` with `.venv` active**:

```bash
# 1) Install chatterbox from source without dependencies (skips pkuseg)
python -m pip install -e . --no-deps

# 2) Install required runtime deps (excluding pkuseg)
python -m pip install \
  "librosa==0.11.0" \
  "transformers==4.46.3" \
  "diffusers==0.29.0" \
  "resemble-perth==1.0.1" \
  "conformer==0.3.2" \
  "safetensors==0.5.3" \
  "s3tokenizer==0.2.0"
```

Then test:

```bash
python - << 'PY'
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cpu")
wav = model.generate("Test Chatterbox TTS on CPU.")
ta.save("test-chatterbox.wav", wav, model.sr)
print("ok")
PY
```

