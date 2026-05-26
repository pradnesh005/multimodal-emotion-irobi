import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

import json, time
import platform, multiprocessing as mp
import warnings

# Silence HF tokenizers fork warning and torchaudio deprecation spam
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    warnings.filterwarnings(
        "ignore",
        message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
        category=UserWarning,
        module="torchaudio._backend.utils"
    )
except Exception:
    pass

# Use spawn start method on macOS to avoid DataLoader deadlocks after forking
try:
    if platform.system() == "Darwin":
        mp.set_start_method("spawn", force=True)
except RuntimeError:
    # start method may already be set
    pass

import whisper
from sentence_transformers import SentenceTransformer

# Configuration
EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SAMPLE_RATE = 16000
SR = SAMPLE_RATE
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multimodal (content + prosody) config
USE_TEXT = True  # set False to train audio-only
WHISPER_MODEL_NAME = "base"
TEXT_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMB_DIM = 384  # all-MiniLM-L6-v2

# Dataset Parsers

def parse_crema_d(data_dir):
    """
    Parses CREMA-D dataset files and returns list of tuples (filepath, emotion_idx, actor_id).
    Filename: <ACTOR4>_<SENTENCE>_<EMOTION>_<INTENSITY>.wav  e.g. 1001_DFA_ANG_XX.wav
    """
    code_to_name = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fearful',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad',
    }
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    samples = []
    for f in files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) < 3:
            continue
        emotion_code = parts[2].upper()
        emotion = code_to_name.get(emotion_code)
        if not (emotion and emotion in EMOTIONS):
            continue
        # Actor is first 4 digits at start of filename
        actor_raw = parts[0]  # e.g., "1001"
        actor_id = f"crema_{actor_raw}"
        emotion_idx = EMOTIONS.index(emotion)
        samples.append((f, emotion_idx, actor_id))
    return samples

def parse_ravdess(data_dir):
    """
    Parses RAVDESS dataset files and returns list of tuples (filepath, emotion_idx, actor_id).
    RAVDESS filename: <Modality>-<VocalChannel>-<Emotion>-<Intensity>-<Statement>-<Repetition>-<Actor>.wav
    Emotion codes: 1=neutral, 2=calm(→neutral), 3=happy, 4=sad, 5=angry, 6=fearful, 7=disgust, 8=surprised
    """
    emotion_map = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'
    }
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    samples = []
    for f in files:
        basename = os.path.basename(f)
        parts = basename.split('-')
        if len(parts) < 7:
            continue
        # Emotion
        try:
            emotion_code = int(parts[2])
        except:
            continue
        emotion = emotion_map.get(emotion_code)
        if not (emotion and emotion in EMOTIONS):
            continue
        # Actor is last field minus .wav, zero-padded 2 digits
        actor_two = os.path.splitext(parts[-1])[0]  # e.g., "01"
        actor_id = f"ravdess_{actor_two}"
        emotion_idx = EMOTIONS.index(emotion)
        samples.append((f, emotion_idx, actor_id))
    return samples

# Dataset Class

class SERDataset(Dataset):
    def __init__(self, samples, augment=False, use_text=USE_TEXT):
        self.samples = samples
        self.augment = augment
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        self.use_text = use_text
        self.text_cache = {}
        if self.use_text:
            # Load once for the dataset. Whisper used for transcription; SentenceTransformer for text embedding.
            try:
                self.whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")
            except Exception as e:
                print(f"[WARN] Failed loading Whisper ({WHISPER_MODEL_NAME}): {e}. Disabling text branch.")
                self.use_text = False
                self.whisper_model = None
            if self.use_text:
                try:
                    self.text_model = SentenceTransformer(TEXT_EMB_MODEL_NAME, device="cpu")
                except Exception as e:
                    print(f"[WARN] Failed loading text embedding model ({TEXT_EMB_MODEL_NAME}): {e}. Disabling text branch.")
                    self.use_text = False
                    self.text_model = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label, _ = self.samples[idx]
        waveform, sr = torchaudio.load(filepath)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Optional augmentation
        if self.augment:
            waveform = self.apply_augmentation(waveform)
        # Placeholder for MCycleGAN preprocessing (if available)
        # waveform = self.apply_mcyclegan(waveform)
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        # Normalize mel spectrogram
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9)

        # SpecAugment (applied only during training/augmentation)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                mel_spec_db = T.FrequencyMasking(freq_mask_param=8)(mel_spec_db)
            if torch.rand(1).item() < 0.5:
                mel_spec_db = T.TimeMasking(time_mask_param=20)(mel_spec_db)

        # --- Ensure fixed time length for batching ---
        # Pad or trim along the time dimension to T_MAX frames
        T_MAX = 300  # you can tune this (e.g., 400)
        cur_T = mel_spec_db.size(-1)
        if cur_T < T_MAX:
            pad_amount = T_MAX - cur_T
            # pad format is (left, right) on the last dimension
            mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, pad_amount))
        else:
            mel_spec_db = mel_spec_db[:, :, :T_MAX]
        # ---------------------------------------------

        # Optional text embedding via Whisper + SentenceTransformer (cached per file)
        if self.use_text:
            if filepath in self.text_cache:
                text_emb = self.text_cache[filepath]
            else:
                try:
                    # Transcribe with Whisper
                    result = self.whisper_model.transcribe(filepath, language="en", fp16=False)
                    text = (result.get("text") or "").strip()
                except Exception as e:
                    print(f"[WARN] Whisper transcription failed for {os.path.basename(filepath)}: {e}")
                    text = ""
                try:
                    vec = self.text_model.encode(text, normalize_embeddings=True)
                except Exception as e:
                    print(f"[WARN] Text embedding failed for {os.path.basename(filepath)}: {e}")
                    vec = np.zeros((TEXT_EMB_DIM,), dtype=np.float32)
                text_emb = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)  # (1, D)
                self.text_cache[filepath] = text_emb
            return mel_spec_db, label, text_emb
        else:
            return mel_spec_db, label

    def apply_augmentation(self, waveform):
        # Example augmentations: Add noise, time shifting
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        if torch.rand(1).item() < 0.5:
            shift = int(torch.randint(-1000, 1000, (1,)).item())
            waveform = torch.roll(waveform, shifts=shift, dims=-1)
        return waveform

    def apply_mcyclegan(self, waveform):
        # Placeholder for MCycleGAN preprocessing
        # For now, return waveform unchanged
        return waveform

# Model Definition

class CRNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=len(EMOTIONS), text_dim=TEXT_EMB_DIM, use_text=USE_TEXT):
        super(CRNN, self).__init__()
        self.use_text = use_text
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (B,16,N_MELS,T)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )
        # After 3x pool, mel dim shrinks by 8×
        cnn_output_dim = 64 * (n_mels // 8)
        self.rnn = nn.GRU(input_size=cnn_output_dim, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(256*2, 1)  # attention over time
        # Project audio context (512 -> 128)
        self.audio_proj = nn.Sequential(
            nn.Linear(256*2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        if self.use_text:
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.classifier = nn.Linear(128 + 128, n_classes)
        else:
            self.classifier = nn.Linear(128, n_classes)

    def forward(self, x, text_emb=None):
        # x: (B,1,N_MELS,T); text_emb: (B, D) or (B,1,D)
        x = self.cnn(x)
        batch, channels, n_mels, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * n_mels)

        rnn_out, _ = self.rnn(x)                      # (B, T', 512) because bidirectional GRU
        attn_scores = self.attn(rnn_out).squeeze(-1)  # (B, T')
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_out).squeeze(1)  # (B, 512)

        a_feat = self.audio_proj(context)  # (B, 128)

        # IMPORTANT:
        # If the model was created with `use_text=True`, then `self.classifier` expects 256-dim input
        # (audio 128 + text 128). During inference you might not provide `text_emb`.
        # In that case we feed a zero text feature so dimensions still match.
        if self.use_text:
            if text_emb is None:
                # Create a zero text embedding on the correct device
                text_emb = torch.zeros((batch, TEXT_EMB_DIM), dtype=a_feat.dtype, device=a_feat.device)
            else:
                # Allow (B,1,D) -> (B,D)
                if text_emb.dim() == 3 and text_emb.size(1) == 1:
                    text_emb = text_emb.squeeze(1)
                # Ensure 2D
                if text_emb.dim() != 2:
                    raise ValueError(f"text_emb must be (B,D) or (B,1,D). Got shape: {tuple(text_emb.shape)}")

            t_feat = self.text_proj(text_emb)  # (B, 128)
            fused = torch.cat([a_feat, t_feat], dim=1)  # (B, 256)
            logits = self.classifier(fused)
        else:
            logits = self.classifier(a_feat)

        return logits

# Helper functions for inference and preprocessing

def mel_spectrogram(waveform):
    """
    Compute normalized log-mel spectrogram for a given waveform tensor.
    Args:
        waveform (Tensor): Tensor of shape (1, N)
    Returns:
        Tensor: Normalized log-mel spectrogram of shape (1, N_MELS, T)
    """
    mel_spec_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    amplitude_to_db = T.AmplitudeToDB()
    mel_spec = mel_spec_transform(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9)
    return mel_spec_db

def trim_or_pad(mel_spec_db, max_len=300):
    """
    Trim or pad the spectrogram to a fixed length (max_len) along the time dimension.
    Args:
        mel_spec_db (Tensor): Tensor of shape (1, N_MELS, T)
        max_len (int): Desired length along time dimension
    Returns:
        Tensor: Tensor of shape (1, N_MELS, max_len)
    """
    cur_len = mel_spec_db.size(-1)
    if cur_len < max_len:
        pad_amount = max_len - cur_len
        mel_spec_db = torch.nn.functional.pad(mel_spec_db, (0, pad_amount))
    else:
        mel_spec_db = mel_spec_db[:, :, :max_len]
    return mel_spec_db

# Training and Evaluation

def train_epoch(model, dataloader, criterion, optimizer, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if len(batch) == 3:
            inputs, labels, text_emb = batch
            text_emb = text_emb.squeeze(1).to(DEVICE)  # (B, D)
        else:
            inputs, labels = batch
            text_emb = None
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs, text_emb=text_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # OneCycleLR (or any per-batch scheduler) step
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass
        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def eval_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if len(batch) == 3:
                inputs, labels, text_emb = batch
                text_emb = text_emb.squeeze(1).to(DEVICE)
            else:
                inputs, labels = batch
                text_emb = None
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs, text_emb=text_emb)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)

# Main Entrypoint

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CRNN-based SER model")
    parser.add_argument("--crema_dir", type=str, required=True, help="Path to CREMA-D dataset")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory to save models")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--early_patience", type=int, default=10, help="Early stopping patience (epochs without val loss improvement)")
    parser.add_argument("--scheduler_factor", type=float, default=0.5, help="LR Reduce factor on plateau")
    parser.add_argument("--scheduler_patience", type=int, default=3, help="LR scheduler patience (epochs without val loss improvement)")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum LR for scheduler")
    parser.add_argument("--max_lr", type=float, default=1e-3, help="OneCycleLR: peak learning rate")
    parser.add_argument("--pct_start", type=float, default=0.3, help="OneCycleLR: percentage of cycle spent increasing LR")
    parser.add_argument("--anneal_strategy", type=str, default="cos", choices=["cos", "linear"], help="OneCycleLR: annealing strategy")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Parsing datasets...")
    crema_samples = parse_crema_d(args.crema_dir)
    print(f"Using ONLY CREMA-D dataset")
    print(f"Found {len(crema_samples)} files in CREMA-D dataset")

    # Combine for reporting
    all_samples = crema_samples
    print(f"Total samples (CREMA-D only): {len(all_samples)}")

    # --- Actor-wise split: split by unique actor IDs within each dataset to prevent speaker leakage ---
    def split_by_actor(samples, train_ratio=0.8, seed=42):
        rng = np.random.RandomState(seed)
        # samples are tuples (path, label, actor)
        actors = sorted(list({a for _, _, a in samples}))
        rng.shuffle(actors)
        split_a = int(len(actors) * train_ratio)
        train_actors = set(actors[:split_a])
        train = [s for s in samples if s[2] in train_actors]
        val = [s for s in samples if s[2] not in train_actors]
        return train, val

    crema_train, crema_val = split_by_actor(crema_samples, train_ratio=0.8, seed=42)

    train_samples = crema_train
    val_samples = crema_val

    np.random.shuffle(train_samples)
    np.random.shuffle(val_samples)

    print(f"Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")
    # --------------------------------------------------------------------------

    def _dl_workers():
        # macOS often hangs with multiprocessing workers after tokenizer/whisper usage
        return 0 if platform.system() == "Darwin" else min(4, (os.cpu_count() or 2))

    train_dataset = SERDataset(train_samples, augment=True, use_text=USE_TEXT)
    val_dataset = SERDataset(val_samples, augment=False, use_text=USE_TEXT)

    _workers = _dl_workers()
    _pin = False
    _persist = False if _workers == 0 else True

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=_workers,
        pin_memory=_pin,
        persistent_workers=_persist,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=_workers,
        pin_memory=_pin,
        persistent_workers=_persist,
    )

    model = CRNN(use_text=USE_TEXT, text_dim=TEXT_EMB_DIM).to(DEVICE)

    # Compute class weights from the training split
    class_counts = np.zeros(len(EMOTIONS), dtype=np.int64)
    for _, lbl, _ in train_samples:
        class_counts[lbl] += 1
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    inv_freq = 1.0 / class_counts.astype(np.float64)
    class_weights = inv_freq / inv_freq.sum() * len(EMOTIONS)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print("Class counts (train):", dict(zip(EMOTIONS, class_counts.tolist())))
    print("Class weights:", dict(zip(EMOTIONS, [float(w) for w in class_weights_tensor.cpu().numpy()])))

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.pct_start,
        anneal_strategy=args.anneal_strategy,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # === Metric trackers and initialization ===
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_epoch = 0
    best_snapshot = {}
    run_started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss, val_acc, val_preds, val_true = eval_epoch(model, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        # --- Track metrics history ---
        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        # Lightweight per-class report each epoch
        labels = list(range(len(EMOTIONS)))
        cm = confusion_matrix(val_true, val_preds, labels=labels)
        print("Confusion matrix:\n", cm)
        # Always pass `labels` so sklearn doesn't drop missing classes (fixes: Number of classes mismatch)
        print(classification_report(
            val_true,
            val_preds,
            labels=labels,
            target_names=EMOTIONS,
            digits=3,
            zero_division=0,
        ))

        # Save best model and snapshot if val_acc improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            best_snapshot = {
                "epoch": int(best_epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "model_path": save_path,
            }
            print(f"Saved best model to {save_path}")

        # Print current learning rate
        current_lr = next(param_group["lr"] for param_group in optimizer.param_groups)
        print(f"Current LR: {current_lr:.6f}")

    # === Evaluate best checkpoint on validation set for a definitive report ===
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    import pandas as pd
    import matplotlib.pyplot as plt

    # Reload best checkpoint if we have it
    best_model_path = best_snapshot.get("model_path", "")
    if isinstance(best_model_path, str) and os.path.isfile(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        model.to(DEVICE)
    else:
        print("[WARN] No best model checkpoint found on disk; using last-epoch weights for final report.")

    # Ensure output dir exists
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    # Run a clean eval to capture preds/labels for report
    val_loss_best, val_acc_best, val_preds_best, val_true_best = eval_epoch(model, val_loader, criterion)

    # === Per-emotion distribution (percentages) on validation set ===
    # Predicted distribution: what emotions the model outputs
    pred_counts = np.bincount(val_preds_best, minlength=len(EMOTIONS)).astype(np.int64)
    pred_pct = (pred_counts / max(int(pred_counts.sum()), 1)) * 100.0

    # True distribution: ground-truth label balance in the validation split
    true_counts = np.bincount(val_true_best, minlength=len(EMOTIONS)).astype(np.int64)
    true_pct = (true_counts / max(int(true_counts.sum()), 1)) * 100.0

    # Print a clear table to terminal
    print("\n=== Per-Emotion Distribution (Validation) ===")
    print("Emotion\t\tPred%\tPred#\tTrue%\tTrue#")
    for i, emo in enumerate(EMOTIONS):
        emo_name = emo.ljust(9)
        print(f"{emo_name}\t{pred_pct[i]:6.2f}%\t{pred_counts[i]:5d}\t{true_pct[i]:6.2f}%\t{true_counts[i]:5d}")

    # Save distribution to CSV for reports
    dist_csv_path = os.path.join(outdir, "emotion_distribution.csv")
    try:
        import csv
        with open(dist_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["emotion", "pred_percent", "pred_count", "true_percent", "true_count"])
            for i, emo in enumerate(EMOTIONS):
                w.writerow([emo, float(pred_pct[i]), int(pred_counts[i]), float(true_pct[i]), int(true_counts[i])])
        print(f"Saved emotion distribution CSV to: {dist_csv_path}")
    except Exception as e:
        print(f"[WARN] Could not write emotion distribution CSV: {e}")
    labels = list(range(len(EMOTIONS)))
    cm_best = confusion_matrix(val_true_best, val_preds_best, labels=labels)
    cls_report = classification_report(
        val_true_best,
        val_preds_best,
        labels=labels,
        target_names=EMOTIONS,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    # Also compute macro averages explicitly
    prec, rec, f1, support = precision_recall_fscore_support(val_true_best, val_preds_best, average=None, labels=list(range(len(EMOTIONS))))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(val_true_best, val_preds_best, average="macro")
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(val_true_best, val_preds_best, average="weighted")


    # Save confusion matrix and per-class metrics to CSV
    cm_csv_path = os.path.join(outdir, "confusion_matrix.csv")
    import numpy as _np  # alias to avoid shadowing
    import csv as _csv
    with open(cm_csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow([""] + EMOTIONS)
        for i, row in enumerate(cm_best):
            writer.writerow([EMOTIONS[i]] + list(row.astype(int)))

    # Per-class metrics dataframe
    try:
        import pandas as pd
        per_class_rows = []
        for i, emo in enumerate(EMOTIONS):
            per_class_rows.append({
                "class": emo,
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            })
        per_class_df = pd.DataFrame(per_class_rows)
        per_class_csv = os.path.join(outdir, "per_class_metrics.csv")
        per_class_df.to_csv(per_class_csv, index=False)
    except Exception as e:
        print(f"[WARN] Could not save per-class metrics CSV: {e}")

    # Save history to CSV
    try:
        hist_csv = os.path.join(outdir, "history.csv")
        import pandas as pd
        pd.DataFrame(history).to_csv(hist_csv, index=False)
    except Exception as e:
        print(f"[WARN] Could not save history CSV: {e}")

    # Plot accuracy and loss curves
    try:
        plt.figure()
        plt.plot(history["epoch"], history["train_acc"], label="train_acc")
        plt.plot(history["epoch"], history["val_acc"], label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy over epochs"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
        acc_plot_path = os.path.join(outdir, "accuracy_curve.png")
        plt.savefig(acc_plot_path, dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(history["epoch"], history["train_loss"], label="train_loss")
        plt.plot(history["epoch"], history["val_loss"], label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss over epochs"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.3)
        loss_plot_path = os.path.join(outdir, "loss_curve.png")
        plt.savefig(loss_plot_path, dpi=160, bbox_inches="tight")
        plt.close()

        # Confusion matrix heatmap
        plt.figure()
        import numpy as np
        im = plt.imshow(cm_best, interpolation="nearest")
        plt.title("Confusion Matrix (Best Checkpoint)")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(EMOTIONS))
        plt.xticks(tick_marks, EMOTIONS, rotation=45, ha="right")
        plt.yticks(tick_marks, EMOTIONS)
        plt.xlabel("Predicted"); plt.ylabel("True")
        for i in range(cm_best.shape[0]):
            for j in range(cm_best.shape[1]):
                plt.text(j, i, f"{cm_best[i, j]}", ha="center", va="center")
        cm_plot_path = os.path.join(outdir, "confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(cm_plot_path, dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[WARN] Could not save plots: {e}")

    # Stash additional best-eval metrics to include in final summary
    best_eval_detail = {
        "val_loss": float(val_loss_best),
        "val_acc": float(val_acc_best),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix_path": cm_csv_path,
        "per_class_metrics_path": os.path.join(outdir, "per_class_metrics.csv"),
        "history_csv_path": os.path.join(outdir, "history.csv"),
        "accuracy_curve_path": acc_plot_path if 'acc_plot_path' in locals() else "",
        "loss_curve_path": loss_plot_path if 'loss_plot_path' in locals() else "",
        "confusion_matrix_plot_path": cm_plot_path if 'cm_plot_path' in locals() else "",
        "emotion_distribution_csv_path": dist_csv_path if 'dist_csv_path' in locals() else "",
        "predicted_emotion_percent": {EMOTIONS[i]: float(pred_pct[i]) for i in range(len(EMOTIONS))} if 'pred_pct' in locals() else {},
    }

    # === After training: save results summary ===
    final_summary = {
        "run_started_at": run_started_at,
        "run_finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": int(args.epochs),
        "best": {
            "epoch": int(best_snapshot.get("epoch", 0)),
            "train_acc": float(best_snapshot.get("train_acc", 0.0)),
            "val_acc": float(best_snapshot.get("val_acc", 0.0)),
            "train_loss": float(best_snapshot.get("train_loss", 0.0)),
            "val_loss": float(best_snapshot.get("val_loss", 0.0)),
            "model_path": best_snapshot.get("model_path", ""),
        },
        "best_eval_detail": best_eval_detail,
        "history": history,
        "config": {
            "crema_dir": args.crema_dir,
            "output_dir": args.output_dir,
            "epochs": args.epochs,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "use_text": USE_TEXT,
            "text_model": TEXT_EMB_MODEL_NAME if USE_TEXT else None,
            "whisper_model": WHISPER_MODEL_NAME if USE_TEXT else None,
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
        },
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    best_pct = final_summary["best"]["val_acc"] * 100.0
    print(f"\n=== Training complete ===")
    print(f"Best epoch: {best_epoch} | Val Acc: {best_pct:.2f}% | Model: {final_summary['best']['model_path']}")
    print(f"Full metrics saved to: {results_path}")

    # === Write a human-readable Markdown report ===
    report_md = os.path.join(args.output_dir, "report.md")
    try:
        with open(report_md, "w") as rf:
            rf.write("# SER Training Report\n\n")
            rf.write(f"- Started: {run_started_at}\n")
            rf.write(f"- Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            rf.write(f"- Epochs: {args.epochs}\n")
            rf.write(f"- Best Epoch: {best_snapshot.get('epoch', best_epoch)}\n")
            rf.write(f"- Best Val Acc: {best_eval_detail.get('val_acc', best_snapshot.get('val_acc', 0.0)) * 100:.2f}%\n")
            rf.write(f"- Model Path: `{best_snapshot.get('model_path','')}`\n")
            rf.write(f"- Use Text: {USE_TEXT}\n")
            if USE_TEXT:
                rf.write(f"  - Whisper: {WHISPER_MODEL_NAME}\n")
                rf.write(f"  - Text Model: {TEXT_EMB_MODEL_NAME}\n")
            rf.write("\n## Curves\n")
            if best_eval_detail.get("accuracy_curve_path"):
                rf.write(f"![Accuracy]({os.path.basename(best_eval_detail['accuracy_curve_path'])})\n\n")
            if best_eval_detail.get("loss_curve_path"):
                rf.write(f"![Loss]({os.path.basename(best_eval_detail['loss_curve_path'])})\n\n")
            rf.write("## Confusion Matrix\n")
            if best_eval_detail.get("confusion_matrix_plot_path"):
                rf.write(f"![Confusion Matrix]({os.path.basename(best_eval_detail['confusion_matrix_plot_path'])})\n\n")
            rf.write("## Emotion Distribution (Validation)\n")
            if best_eval_detail.get("emotion_distribution_csv_path"):
                rf.write(f"- CSV: `{os.path.basename(best_eval_detail['emotion_distribution_csv_path'])}`\n")
            pred_dist = best_eval_detail.get("predicted_emotion_percent", {})
            if pred_dist:
                rf.write("\n**Predicted emotion percentages:**\n\n")
                for emo in EMOTIONS:
                    if emo in pred_dist:
                        rf.write(f"- {emo}: {pred_dist[emo]:.2f}%\n")
            rf.write("\n")
            rf.write("## Per-Class Metrics\n")
            rf.write(f"- CSV: `{os.path.basename(best_eval_detail.get('per_class_metrics_path',''))}`\n")
            rf.write("\n## History\n")
            rf.write(f"- CSV: `{os.path.basename(best_eval_detail.get('history_csv_path',''))}`\n")
            rf.write("\n## Raw Summary JSON\n")
            rf.write(f"- File: `{os.path.basename(results_path)}`\n")
        print(f"Detailed Markdown report written to: {report_md}")
    except Exception as e:
        print(f"[WARN] Could not write Markdown report: {e}")
