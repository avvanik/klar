#!/usr/bin/env python3
"""Generate the sample recordings with ElevenLabs Text to Speech.

Why this exists: the repo commits no audio (see .gitignore). Run this once to
synthesize three short two-speaker clips (English, German, Spanish) that match
samples/expectations.yaml, so `python -m klar eval` has ground truth to check.

Usage:
    export ELEVENLABS_API_KEY=...     # or put it in .env
    python scripts/generate_samples.py

Requires: pip install elevenlabs python-dotenv

Approach: each line is synthesized with one of two distinct voices, then the
per-line MP3 segments are concatenated. Two voices => Scribe's diarization sees
real speaker turns. No ffmpeg needed (CBR MP3 frames concatenate cleanly).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
TTS_MODEL = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"

# (filename, [(speaker_slot, text), ...]) - speaker_slot 0/1 picks the voice.
DIALOGUES: list[tuple[str, list[tuple[int, str]]]] = [
    (
        "en_sales_call.mp3",
        [
            (0, "Hi, thanks for making time. I'm calling from Acme about your proof of concept."),
            (1, "Of course. We were impressed, but we still need pricing before we commit."),
            (0, "Understood. I'll send the order form and pricing by Friday."),
            (1, "Great. And can you include the security review document as well?"),
            (0, "Yes. I'll loop in our security team and get you that document this week."),
            (1, "Perfect. If the numbers work, we'll sign off on the proof of concept."),
        ],
    ),
    (
        "de_support_call.mp3",
        [
            (0, "Support, guten Tag. Wie kann ich Ihnen helfen?"),
            (1, "Mein CardioScan Gerät startet nach dem letzten Firmware-Update nicht mehr."),
            (0, "Das tut mir leid. Haben Sie nach dem Firmware-Update eine Sensor-Kalibrierung durchgeführt?"),
            (1, "Nein, das wusste ich nicht. Ist das Gerät noch in der Garantie?"),
            (0, "Ich prüfe das und sende Ihnen heute eine Anleitung zur Sensor-Kalibrierung."),
            (1, "Vielen Dank, das beruhigt mich."),
        ],
    ),
    (
        "es_project_sync.mp3",
        [
            (0, "Hola equipo, vamos a revisar el estado del proyecto de diseño."),
            (1, "El diseño va bien, pero necesitamos aprobar el presupuesto esta semana."),
            (0, "De acuerdo. Enviaré el presupuesto actualizado al cliente mañana."),
            (1, "Perfecto. ¿Quién se encarga de la revisión final del diseño?"),
            (0, "Yo me encargo de la revisión final del diseño antes del viernes."),
        ],
    ),
    (
        "fr_kickoff_call.mp3",
        [
            (0, "Bonjour, merci pour votre temps. Je vous appelle au sujet du budget du nouveau projet."),
            (1, "Bien sûr. Nous sommes intéressés, mais nous devons valider le calendrier avant de nous engager."),
            (0, "Je comprends. Je vous enverrai le devis et le calendrier avant vendredi."),
            (1, "Parfait. Pouvez-vous aussi inclure le plan de formation ?"),
            (0, "Oui, j'ajouterai le plan de formation au document cette semaine."),
        ],
    ),
]


def pick_two_voices(client) -> list[str]:
    resp = client.voices.get_all()
    voices = getattr(resp, "voices", resp)
    ids = [getattr(v, "voice_id", None) or v.get("voice_id") for v in voices]
    ids = [i for i in ids if i]
    if len(ids) < 2:
        raise SystemExit("Need at least 2 voices on this account to make two speakers.")
    return ids[:2]


def synth_line(client, voice_id: str, text: str) -> bytes:
    audio = client.text_to_speech.convert(
        voice_id=voice_id, text=text, model_id=TTS_MODEL, output_format=OUTPUT_FORMAT,
    )
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    return b"".join(audio)  # SDK returns a byte-chunk iterator


def main() -> int:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY not set (see .env.example).", file=sys.stderr)
        return 2
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError:
        print("pip install elevenlabs", file=sys.stderr)
        return 2

    client = ElevenLabs(api_key=api_key)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    voices = pick_two_voices(client)
    print(f"Using voices: {voices}")

    for fname, lines in DIALOGUES:
        out = SAMPLES_DIR / fname
        print(f"Synthesizing {fname} ({len(lines)} lines) ...")
        with open(out, "wb") as fh:
            for slot, text in lines:
                fh.write(synth_line(client, voices[slot % 2], text))
        print(f"  wrote {out} ({out.stat().st_size} bytes)")

    print("\nDone. Now run:  python -m klar eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
