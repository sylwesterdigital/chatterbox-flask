import io
import logging
import os
import tempfile
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio as ta
from flask import Flask, jsonify, render_template, request, send_file

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = app.logger

# ───────────────────────── Chatterbox config ─────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL: Optional[ChatterboxMultilingualTTS] = None

# Default demo text + reference clips per language (from upstream demo)
LANGUAGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}


def default_audio_for_ui(lang: str) -> Optional[str]:
    """Return default reference audio URL for a language, if any."""
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def default_text_for_ui(lang: str) -> str:
    """Return default demo text for a language, or empty string."""
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_supported_languages_for_template() -> list[Dict[str, Any]]:
    """Return supported languages with code, display name, and default flags."""
    out: list[Dict[str, Any]] = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        out.append(
            {
                "code": code,
                "name": name,
                "has_sample": bool(default_audio_for_ui(code)),
                "default_text": default_text_for_ui(code),
            }
        )
    return out


def get_or_load_model() -> ChatterboxMultilingualTTS:
    """Load multilingual Chatterbox model once and keep it in memory."""
    global MODEL
    if MODEL is None:
        log.info("Loading ChatterboxMultilingualTTS on %s", DEVICE)
        MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        if hasattr(MODEL, "to") and getattr(MODEL, "device", None) != DEVICE:
            MODEL.to(DEVICE)
        log.info("Chatterbox model ready (device=%s, sr=%s)", getattr(MODEL, "device", "n/a"), MODEL.sr)
    return MODEL


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible output when seed != 0."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def resolve_audio_prompt(language_id: str, uploaded_path: Optional[str], use_default: bool) -> Optional[str]:
    """Decide which reference audio to use: uploaded file or language default or none."""
    if uploaded_path:
        return uploaded_path
    if use_default:
        return default_audio_for_ui(language_id)
    return None


def generate_tts(
    text: str,
    language_id: str,
    audio_prompt_path: Optional[str],
    exaggeration: float,
    temperature: float,
    cfg_weight: float,
    seed: int,
) -> Tuple[int, np.ndarray]:
    """Generate waveform (sr, mono wav) using Chatterbox multilingual backend."""
    model = get_or_load_model()

    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("Empty text.")
    clean_text = clean_text[:300]

    if seed:
        set_seed(int(seed))

    generate_kwargs: Dict[str, Any] = {
        "exaggeration": float(exaggeration),
        "temperature": float(temperature),
        "cfg_weight": float(cfg_weight),
    }
    if audio_prompt_path:
        generate_kwargs["audio_prompt_path"] = audio_prompt_path

    log.info(
        "TTS: lang=%s, len=%d, exaggeration=%.3f, temp=%.3f, cfg=%.3f, prompt=%s",
        language_id,
        len(clean_text),
        exaggeration,
        temperature,
        cfg_weight,
        "yes" if audio_prompt_path else "no",
    )

    wav = model.generate(clean_text, language_id=language_id, **generate_kwargs)
    mono = wav.squeeze(0).detach().cpu().numpy()
    return model.sr, mono


# ───────────────────────────── Routes ─────────────────────────────

@app.get("/")
def index():
    """Render main Chatterbox UI."""
    languages = get_supported_languages_for_template()
    # Pick a default language that exists in the model
    default_lang = "en" if "en" in SUPPORTED_LANGUAGES else languages[0]["code"]
    return render_template(
        "index.html",
        languages=languages,
        default_lang=default_lang,
    )


@app.post("/api/tts")
def api_tts():
    """Generate TTS audio from posted form data and return as WAV."""
    try:
        # Support both JSON and form; UI uses form with optional file upload
        if request.is_json:
            data = request.get_json(silent=True) or {}
            text = data.get("text", "")
            language_id = data.get("language_id", "en")
            exaggeration = float(data.get("exaggeration", 0.5))
            temperature = float(data.get("temperature", 0.8))
            cfg_weight = float(data.get("cfg_weight", 0.5))
            seed = int(data.get("seed", 0))
            use_default_prompt = bool(data.get("use_default_prompt", False))
            uploaded_path = None
        else:
            form = request.form
            text = form.get("text", "")
            language_id = form.get("language_id", "en")
            exaggeration = float(form.get("exaggeration", 0.5))
            temperature = float(form.get("temperature", 0.8))
            cfg_weight = float(form.get("cfg_weight", 0.5))
            seed = int(form.get("seed", 0) or 0)
            use_default_prompt = form.get("use_default_prompt", "0") in ("1", "true", "on", "yes")

            uploaded_path = None
            file = request.files.get("ref_audio")
            if file and file.filename:
                suffix = os.path.splitext(file.filename)[1] or ".wav"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                try:
                    file.save(tmp.name)
                    uploaded_path = tmp.name
                finally:
                    tmp.close()

        if language_id not in SUPPORTED_LANGUAGES:
            return jsonify({"error": f"Unsupported language_id '{language_id}'"}), 400

        audio_prompt_path = resolve_audio_prompt(language_id, uploaded_path, use_default_prompt)

        sr, wav = generate_tts(
            text=text,
            language_id=language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfg_weight,
            seed=seed,
        )

        buf = io.BytesIO()
        tensor = torch.from_numpy(wav).unsqueeze(0).to(torch.float32)
        ta.save(buf, tensor, sr, format="WAV")
        buf.seek(0)

        if uploaded_path and os.path.exists(uploaded_path):
            try:
                os.unlink(uploaded_path)
            except Exception:
                log.exception("Failed to remove temp file %s", uploaded_path)

        return send_file(
            buf,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="chatterbox-tts.wav",
        )

    except Exception as e:
        log.exception("TTS generation failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_RUN_PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug, use_reloader=True)
