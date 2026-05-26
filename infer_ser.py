# infer_ser.py
import os
import numpy as np
import torch
import sounddevice as sd
import librosa
import soundfile as sf
import tempfile
import time
from collections import deque
import cv2

try:
    import whisper
    _WHISPER_AVAILABLE = True
except ImportError:
    whisper = None
    _WHISPER_AVAILABLE = False

try:
    from deepface import DeepFace
    _DEEPFACE_AVAILABLE = True
except Exception:
    DeepFace = None
    _DEEPFACE_AVAILABLE = False


from train_ser import CRNN, mel_spectrogram, trim_or_pad, EMOTIONS

# Final emotion classes used for live multimodal output.
# The trained tone model still predicts the original EMOTIONS list from train_ser.py,
# but similar/ambiguous classes are merged for stable real-time fusion.
FINAL_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]

MERGE_TO_FINAL = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fearful",
    "disgust": "angry",
    "surprised": "surprised",
}


def collapse_to_final_emotions(probs):
    final = {e: 0.0 for e in FINAL_EMOTIONS}
    for emo, value in probs.items():
        mapped = MERGE_TO_FINAL.get(emo)
        if mapped in final:
            final[mapped] += float(value)

    total = sum(final.values())
    if total > 0:
        final = {k: v / total for k, v in final.items()}
    else:
        final = {e: 1.0 / len(FINAL_EMOTIONS) for e in FINAL_EMOTIONS}

    return final

# ---------- CONFIG ----------
SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Fusion weights (updated to prioritize facial emotion)
# Face has the highest priority because it is the most stable modality
# Tone weight reduced because it was dominating final predictions
W_FACE = 0.50
W_TONE = 0.20
W_TEXT = 0.30

# ---------- WHISPER (Speech-to-text) ----------
_whisper_model = None
def transcribe_whisper(y_np, sr=SR, model_name="base"):
    global _whisper_model
    if not _WHISPER_AVAILABLE:
        return None
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, y_np, sr)
        result = _whisper_model.transcribe(tmp.name, fp16=False, language="en")
    text = (result.get("text") or "").strip()
    return text if text else None

# ---------- TEXT EMOTION MODEL ----------
# Uses pretrained classifier: j-hartmann/emotion-english-distilroberta-base
# Outputs: anger, disgust, fear, joy, neutral, sadness, surprise
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_text_tokenizer = None
_text_model = None

TEXT_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
TEXT_LABEL_MAP = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprised",
}

FACE_LABEL_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprised",
    "neutral": "neutral",
}

def load_text_emotion_model(device=DEVICE):
    global _text_tokenizer, _text_model
    if not _TRANSFORMERS_AVAILABLE:
        return None, None
    if _text_model is None or _text_tokenizer is None:
        _text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        _text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME)
        _text_model.to(device)
        _text_model.eval()
    return _text_tokenizer, _text_model

def predict_text_emotion_probs(text, device=DEVICE):
    """
    Returns:
      probs_dict: {EMOTIONS[i]: prob}
    """
    # Default: uniform if no text model / no text
    if not _TRANSFORMERS_AVAILABLE or not text or len(text.strip()) == 0:
        return {e: 1.0/len(FINAL_EMOTIONS) for e in FINAL_EMOTIONS}

    tok, mdl = load_text_emotion_model(device=device)
    if tok is None or mdl is None:
        return {e: 1.0/len(FINAL_EMOTIONS) for e in FINAL_EMOTIONS}

    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Model labels -> our EMOTIONS list
    id2label = mdl.config.id2label  # e.g. {0:"anger",...}
    mapped = {e: 0.0 for e in EMOTIONS}

    for idx, p in enumerate(probs):
        lab = id2label[idx].lower()
        if lab in TEXT_LABEL_MAP:
            mapped_emo = TEXT_LABEL_MAP[lab]
            if mapped_emo in mapped:
                mapped[mapped_emo] += float(p)

    # Handle "calm" which doesn't exist in text model: set it to 0 and renormalize
    mapped["calm"] = 0.0
    s = sum(mapped.values())
    if s > 0:
        for k in mapped:
            mapped[k] /= s
    else:
        mapped = {e: 1.0/len(FINAL_EMOTIONS) for e in FINAL_EMOTIONS}

    return collapse_to_final_emotions(mapped)

def uniform_face_probs():
    probs = {e: 0.0 for e in FINAL_EMOTIONS}
    probs["neutral"] = 1.0
    return probs

# Helper: zero probabilities for all emotions (used when no speech detected)
def zero_emotion_probs():
    return {e: 0.0 for e in FINAL_EMOTIONS}

def is_uniform_neutral_face(probs):
    return probs.get("neutral", 0.0) >= 0.999 and sum(v for k, v in probs.items() if k != "neutral") <= 1e-9


def detect_largest_face(frame_bgr):
    if frame_bgr is None:
        return None, None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None, None
    # choose the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    return frame_bgr[y:y+h, x:x+w], (x, y, w, h)


def expand_face_box(frame_shape, box, pad=0.20):
    h_img, w_img = frame_shape[:2]
    x, y, w, h = box
    px = int(w * pad)
    py = int(h * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(w_img, x + w + px)
    y2 = min(h_img, y + h + py)
    return x1, y1, x2, y2


def capture_face_frame(camera_index=0):
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None
        # warm up camera
        for _ in range(5):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return frame
    except Exception:
        return None


def predict_face_emotion_probs(frame_bgr):
    if frame_bgr is None:
        return None, "face frame unavailable", None
    if not _DEEPFACE_AVAILABLE:
        return None, "deepface not installed/import failed", None

    try:
        _, face_box = detect_largest_face(frame_bgr)
        if face_box is None:
            return None, "no face detected", None

        x1, y1, x2, y2 = expand_face_box(frame_bgr.shape, face_box, pad=0.20)
        face_crop_bgr = frame_bgr[y1:y2, x1:x2]
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return None, "empty face crop", None

        # DeepFace works better if we pass the cropped face and skip its own detector.
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)

        result = DeepFace.analyze(
            img_path=face_rgb,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True,
        )

        if isinstance(result, list):
            result = result[0]

        raw_emotions = result.get("emotion", {})
        mapped = {e: 0.0 for e in EMOTIONS}

        for raw_label, score in raw_emotions.items():
            mapped_label = FACE_LABEL_MAP.get(str(raw_label).lower())
            if mapped_label in mapped:
                mapped[mapped_label] += float(score) / 100.0

        mapped["calm"] = 0.0
        total = sum(mapped.values())
        if total > 0:
            for k in mapped:
                mapped[k] /= total
            return collapse_to_final_emotions(mapped), "face detected", (x1, y1, x2, y2)

        return None, "no emotion scores returned", (x1, y1, x2, y2)
    except Exception as e:
        return None, f"face error: {str(e)[:80]}", None

# ---------- TONE MODEL ----------
def load_tone_model(ckpt_path, device=DEVICE):
    """
    Load model EXACTLY as it was trained.
    Stage 1 checkpoint was trained with text branch present,
    so we must instantiate CRNN with default settings (no forcing use_text=False).
    """
    model = CRNN()  # match training architecture
    state = torch.load(ckpt_path, map_location=device)

    # Load non-strict to avoid minor unused text layers during inference
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model

def predict_tone_probs(model, y_np, device=DEVICE):
    """
    y_np: mono numpy audio
    Returns:
      probs_dict: {EMOTIONS[i]: prob}
    """
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(0)  # (1, N)
    logm = mel_spectrogram(y)                # (1, n_mels, T)
    logm = trim_or_pad(logm, max_len=300)    # fixed time
    x = logm.unsqueeze(0).to(device)         # (1, 1, n_mels, T)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        # Confidence smoothing to reduce extreme tone dominance
        probs = probs * 0.75
        probs = probs / probs.sum()
    raw_probs = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
    return collapse_to_final_emotions(raw_probs)

# ---------- FUSION ----------
def fuse_probs(face_probs, tone_probs, text_probs, w_face=W_FACE, w_tone=W_TONE, w_text=W_TEXT):
    fused = {}

    text_sum = sum(text_probs.values()) if text_probs else 0.0
    no_valid_speech = text_sum <= 1e-9

    # If no speech is detected, do not allow silence/noise-based tone or empty speech
    # to affect the final emotion. Use face as the reliable real-time modality.
    if no_valid_speech:
        adaptive_face = 1.0
        adaptive_tone = 0.0
        adaptive_text = 0.0
    else:
        adaptive_face = w_face
        adaptive_tone = w_tone
        adaptive_text = w_text

    for e in FINAL_EMOTIONS:
        fused[e] = (
            adaptive_face * face_probs.get(e, 0.0)
            + adaptive_tone * tone_probs.get(e, 0.0)
            + adaptive_text * text_probs.get(e, 0.0)
        )

    s = sum(fused.values())
    if s > 0:
        for e in fused:
            fused[e] /= s
    else:
        fused = {e: 1.0 / len(FINAL_EMOTIONS) for e in FINAL_EMOTIONS}

    return fused

def topk(dist, k=3):
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    return items[:k]

# ---------- MAIN ----------
def record_audio(duration=3.0):
    audio = sd.rec(int(duration * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()

# --------- LIVE AUDIO STREAM HELPERS ----------
def create_live_audio_buffer(max_seconds=10):
    return deque(maxlen=int(SR * max_seconds))


def make_audio_callback(audio_buffer):
    def callback(indata, frames, time_info, status):
        if status:
            pass
        mono = indata[:, 0].copy()
        audio_buffer.extend(mono.tolist())
    return callback


def get_latest_audio_window(audio_buffer, duration=3.0):
    required = int(SR * duration)
    if len(audio_buffer) < required:
        return None
    data = list(audio_buffer)[-required:]
    return np.array(data, dtype=np.float32)

def get_top_label_and_score(probs):
    top = max(probs, key=probs.get)
    return top, probs[top]


def draw_status_panel(frame, face_probs, tone_probs, text_probs, fused_probs, transcript_display, face_status, face_box=None):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    if face_box is not None:
        x1, y1, x2, y2 = face_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    panel_h = 190
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    alpha = 0.60
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    face_top, face_score = get_top_label_and_score(face_probs)
    tone_top, tone_score = get_top_label_and_score(tone_probs)
    text_top, text_score = get_top_label_and_score(text_probs)
    fused_top, fused_score = get_top_label_and_score(fused_probs)

    y0 = h - panel_h + 28
    dy = 28
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, f"Face: {face_top} ({face_score*100:.1f}%)", (15, y0), font, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Tone: {tone_top} ({tone_score*100:.1f}%)", (15, y0 + dy), font, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Speech: {text_top} ({text_score*100:.1f}%)", (15, y0 + 2*dy), font, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"Combined: {fused_top} ({fused_score*100:.1f}%)", (15, y0 + 3*dy), font, 0.75, (0, 255, 255), 2)

    transcript_short = transcript_display if len(transcript_display) <= 55 else transcript_display[:52] + "..."
    cv2.putText(frame, f"You said: {transcript_short}", (15, y0 + 4*dy), font, 0.55, (200, 255, 200), 1)

    if face_status:
        status_short = face_status if len(face_status) <= 70 else face_status[:67] + "..."
        cv2.putText(frame, f"Face status: {status_short}", (15, y0 + 5*dy), font, 0.45, (180, 180, 255), 1)

    cv2.putText(frame, "Press q to quit", (w - 150, 25), font, 0.55, (0, 255, 255), 2)
    return frame

def run_continuous_mode(args, tone_model):
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Try a different --camera_index.")

    cv2.namedWindow("Multimodal Emotion Detection", cv2.WINDOW_NORMAL)

    # default states before first prediction
    face_probs = uniform_face_probs()
    tone_probs = {e: 1.0 / len(FINAL_EMOTIONS) for e in FINAL_EMOTIONS}
    text_probs = {e: 1.0 / len(FINAL_EMOTIONS) for e in FINAL_EMOTIONS}
    fused_probs = fuse_probs(face_probs, tone_probs, text_probs)
    transcript_display = "(waiting for first audio window...)"
    face_display = "waiting for face detection"
    face_box = None

    audio_window = 3.0
    last_audio_update = 0.0
    # Update face emotion once every 1 second.
    # Webcam remains fully live because audio runs in a separate non-blocking stream.
    face_update_interval = 1.0
    last_face_update = 0.0

    # Non-blocking microphone stream so camera/face detection can keep updating continuously.
    audio_buffer = create_live_audio_buffer(max_seconds=10)
    audio_stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        dtype="float32",
        callback=make_audio_callback(audio_buffer),
    )
    audio_stream.start()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        now = time.time()

        # Face branch updates continuously and independently from audio.
        if now - last_face_update >= face_update_interval:
            if args.no_face:
                face_probs = uniform_face_probs()
                face_top = max(face_probs, key=face_probs.get)
                face_display = "face disabled"
                face_box = None
            else:
                new_face_probs, face_status, new_face_box = predict_face_emotion_probs(frame_bgr)
                if new_face_box is not None:
                    face_box = new_face_box
                if new_face_probs is not None and not is_uniform_neutral_face(new_face_probs):
                    face_probs = new_face_probs
                    face_top = max(face_probs, key=face_probs.get)
                    face_display = f"{face_status} | {face_top} ({face_probs[face_top]*100:.2f}%)"
                else:
                    face_top = max(face_probs, key=face_probs.get)
                    face_display = f"{face_status} | using last valid: {face_top} ({face_probs[face_top]*100:.2f}%)"
            last_face_update = now

        # Audio/tone/speech branch updates periodically without blocking the camera.
        if now - last_audio_update >= audio_window:
            audio_np = get_latest_audio_window(audio_buffer, duration=audio_window)

            if audio_np is not None:
                # tone branch
                tone_probs = predict_tone_probs(tone_model, audio_np, device=DEVICE)

                # speech branch
                transcript = None
                if not args.no_asr:
                    transcript = transcribe_whisper(audio_np, sr=SR)

                if args.no_asr:
                    transcript_display = "(ASR disabled via --no_asr)"
                    text_probs = zero_emotion_probs()
                elif transcript and transcript.strip():
                    transcript_display = transcript.strip()
                    text_probs = predict_text_emotion_probs(transcript)
                else:
                    transcript_display = "(ASR unavailable / no speech detected)"
                    text_probs = zero_emotion_probs()

                fused_probs = fuse_probs(face_probs, tone_probs, text_probs)

                # console update each audio cycle
                face_top, face_score = get_top_label_and_score(face_probs)
                tone_top, tone_score = get_top_label_and_score(tone_probs)
                text_top, text_score = get_top_label_and_score(text_probs)
                fused_top, fused_score = get_top_label_and_score(fused_probs)
                print("\n==================== LIVE UPDATE ====================")
                print("YOU SAID:", transcript_display)
                print("FACE STATUS:", face_display)
                print(f"FACE     : {face_top} ({face_score*100:.2f}%)")
                print(f"TONE     : {tone_top} ({tone_score*100:.2f}%)")
                print(f"SPEECH   : {text_top} ({text_score*100:.2f}%)")
                print(f"COMBINED : {fused_top} ({fused_score*100:.2f}%)")
                print("====================================================")
            else:
                transcript_display = "(warming up microphone buffer...)"
                fused_probs = fuse_probs(face_probs, tone_probs, text_probs)

            last_audio_update = now

        display = draw_status_panel(
            frame_bgr.copy(),
            face_probs,
            tone_probs,
            text_probs,
            fused_probs,
            transcript_display,
            face_display,
            face_box,
        )
        cv2.imshow("Multimodal Emotion Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    try:
        audio_stream.stop()
        audio_stream.close()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to tone model checkpoint (.pth)")
    ap.add_argument("--mic", action="store_true", help="Use microphone input")
    ap.add_argument("--wav", help="Path to wav file (if not using mic)")
    ap.add_argument("--no_asr", action="store_true", help="Disable speech-to-text")
    ap.add_argument("--no_face", action="store_true", help="Disable face emotion detection")
    ap.add_argument("--camera_index", type=int, default=0, help="Webcam index for face capture")
    args = ap.parse_args()

    tone_model = load_tone_model(args.ckpt, device=DEVICE)

    # Live continuous mode for microphone input
    if args.mic:
        run_continuous_mode(args, tone_model)
        raise SystemExit(0)

    # File-based one-shot mode
    if not args.wav:
        raise SystemExit("For one-shot mode provide --wav <path>, or use --mic for continuous live mode.")
    y_np, _ = librosa.load(args.wav, sr=SR, mono=True)

    # 0) Face prediction
    if args.no_face:
        face_probs = uniform_face_probs()
        face_top = max(face_probs, key=face_probs.get)
        face_display = "face disabled"
    else:
        frame_bgr = capture_face_frame(camera_index=args.camera_index)
        new_face_probs, face_status, _ = predict_face_emotion_probs(frame_bgr)
        if new_face_probs is not None and not is_uniform_neutral_face(new_face_probs):
            face_probs = new_face_probs
        else:
            face_probs = uniform_face_probs()
        face_top = max(face_probs, key=face_probs.get)
        face_display = f"{face_status} | {face_top} ({face_probs[face_top]*100:.2f}%)"

    # 1) Tone prediction
    tone_probs = predict_tone_probs(tone_model, y_np, device=DEVICE)
    tone_top = max(tone_probs, key=tone_probs.get)

    # 2) Speech prediction (ASR + text emotion model)
    transcript = None
    if not args.no_asr:
        transcript = transcribe_whisper(y_np, sr=SR)

    # Always show what the user spoke (if ASR is available)
    if args.no_asr:
        transcript_display = "(ASR disabled via --no_asr)"
    elif transcript and transcript.strip():
        transcript_display = transcript.strip()
    else:
        # Whisper not installed / failed / or no speech detected
        transcript_display = "(ASR unavailable. Install: `pip install -U openai-whisper` and `brew install ffmpeg`)"

    if transcript and transcript.strip():
        text_probs = predict_text_emotion_probs(transcript)
    else:
        text_probs = zero_emotion_probs()
    text_top = max(text_probs, key=text_probs.get)

    # 3) Combined
    fused_probs = fuse_probs(face_probs, tone_probs, text_probs)
    fused_top = max(fused_probs, key=fused_probs.get)

    # ---------- PRINT RESULTS ----------
    print("\n==================== RESULTS ====================")

    print("\nYOU SAID:")
    print(transcript_display)

    print("\n(0) FACE-BASED (Initial visual emotion):")
    print("Top:", face_top, f"({face_probs[face_top]*100:.2f}%)")
    print("Status:", face_display)
    for name, p in topk(face_probs, 3):
        print(f"  {name:<9} {p*100:.2f}%")

    print("\n(1) TONE-BASED (How you spoke):")
    print("Top:", tone_top, f"({tone_probs[tone_top]*100:.2f}%)")
    for name, p in topk(tone_probs, 3):
        print(f"  {name:<9} {p*100:.2f}%")

    print("\n(2) SPEECH-BASED (What you said):")
    print("Transcript:", transcript_display)
    print("Top:", text_top, f"({text_probs[text_top]*100:.2f}%)")
    for name, p in topk(text_probs, 3):
        print(f"  {name:<9} {p*100:.2f}%")

    print("\n(3) COMBINED (Face + Tone + Speech):")
    print("Top:", fused_top, f"({fused_probs[fused_top]*100:.2f}%)")
    for name, p in topk(fused_probs, 3):
        print(f"  {name:<9} {p*100:.2f}%")

    print("\n=================================================")