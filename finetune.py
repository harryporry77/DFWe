import os
import json
import random
import numpy as np
import argparse
import torch
import torch.amp as amp
from functools import partial
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from transformers import AutoModelForAudioClassification, AutoProcessor, get_linear_schedule_with_warmup
from torchaudio.transforms import FrequencyMasking, TimeMasking

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_any_dataset(path):
    try:
        return load_dataset(path)
    except Exception:
        return load_from_disk(path)

def collate_fn(batch, processor, sampling_rate, apply_augmentation=False, aug=None):
    features = processor([x["audio"]["array"] for x in batch], sampling_rate=sampling_rate, return_tensors="pt").input_features
    if apply_augmentation and aug:
        b, f, t = features.shape
        out = []
        for i in range(b):
            x = features[i]
            if random.random() < aug["freq_mask_prob"]:
                x = FrequencyMasking(freq_mask_param=aug["freq_mask_param"])(x.unsqueeze(0)).squeeze(0)
            if random.random() < aug["time_mask_prob"]:
                x = TimeMasking(time_mask_param=aug["time_mask_param"])(x.unsqueeze(0)).squeeze(0)
            if random.random() < aug["amplitude_scale_prob"]:
                scale = random.uniform(aug["amplitude_scale_min"], aug["amplitude_scale_max"])
                x = x * scale
            out.append(x)
        features = torch.stack(out, dim=0)
    return {"input_features": features, "labels": torch.tensor([x["label"] for x in batch])}

def compute_metrics(preds, labels):
    return {"acc": accuracy_score(labels, preds), "uar": recall_score(labels, preds, average="macro"), "war": recall_score(labels, preds, average="weighted"), "f1": f1_score(labels, preds, average="weighted")}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model_id", type=str, default="openai/whisper-tiny")
    p.add_argument("--save_path", type=str, default="./outputs/finetune_loso")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--sampling_rate", type=int, default=16000)
    p.add_argument("--unfreeze_last_n_encoder_layers", type=int, default=4)
    p.add_argument("--activation_dropout", type=float, default=0.1)
    p.add_argument("--attention_dropout", type=float, default=0.1)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--encoder_lr", type=float, default=1e-4)
    p.add_argument("--projector_lr", type=float, default=4e-4)
    p.add_argument("--classifier_lr", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--use_data_aug", action="store_true")
    p.add_argument("--freq_mask_param", type=int, default=20)
    p.add_argument("--time_mask_param", type=int, default=35)
    p.add_argument("--freq_mask_prob", type=float, default=0.7)
    p.add_argument("--time_mask_prob", type=float, default=0.7)
    p.add_argument("--amplitude_scale_prob", type=float, default=0.3)
    p.add_argument("--amplitude_scale_min", type=float, default=0.8)
    p.add_argument("--amplitude_scale_max", type=float, default=1.2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--only_fold", type=int, default=-1)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    dataset = load_any_dataset(args.data_path)
    all_data = []
    groups = []
    if isinstance(dataset, DatasetDict):
        sessions = dataset.values()
    else:
        sessions = [dataset]
    for session in sessions:
        for item in session:
            audio_path = item["audio"]["path"]
            speaker_id = os.path.basename(audio_path).split("_")[0]
            item["speaker_id"] = speaker_id
            all_data.append(item)
            groups.append(speaker_id)
    if len(all_data) == 0:
        raise RuntimeError("No data loaded")

    all_labels = [x["label"] for x in all_data]
    all_indices = np.arange(len(all_data))
    logo = LeaveOneGroupOut()
    splits = list(logo.split(all_indices, all_labels, groups))
    fold_indices = range(len(splits)) if args.only_fold < 0 else [args.only_fold]
    processor = AutoProcessor.from_pretrained(args.model_id)

    aug_cfg = {"freq_mask_param": args.freq_mask_param, "time_mask_param": args.time_mask_param, "freq_mask_prob": args.freq_mask_prob, "time_mask_prob": args.time_mask_prob, "amplitude_scale_prob": args.amplitude_scale_prob, "amplitude_scale_min": args.amplitude_scale_min, "amplitude_scale_max": args.amplitude_scale_max}

    fold_results = []
    for fi in fold_indices:
        train_idx, val_idx = splits[fi]
        model_path = os.path.join(args.save_path, f"fold{fi+1}_best_model.pt")
        metrics_path = os.path.join(args.save_path, f"fold{fi+1}_best_metrics.json")
        if not args.force_retrain and os.path.exists(model_path) and os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                fold_results.append(json.load(f))
            continue

        train_ds = Dataset.from_dict({k: [all_data[i][k] for i in train_idx] for k in all_data[0].keys()})
        val_ds = Dataset.from_dict({k: [all_data[i][k] for i in val_idx] for k in all_data[0].keys()})

        model = AutoModelForAudioClassification.from_pretrained(args.model_id, num_labels=args.num_classes, activation_dropout=args.activation_dropout, attention_dropout=args.attention_dropout)
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "classifier"):
            for p in model.classifier.parameters():
                p.requires_grad = True
        if hasattr(model, "projector"):
            for p in model.projector.parameters():
                p.requires_grad = True
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            for p in model.encoder.layers[-args.unfreeze_last_n_encoder_layers:].parameters():
                p.requires_grad = True

        device = torch.device(args.device)
        model = model.to(device)
        use_dp = torch.cuda.is_available() and torch.cuda.device_count() > 1
        if use_dp:
            model = torch.nn.DataParallel(model)

        params_seen = set()
        param_groups = []
        if isinstance(model, torch.nn.DataParallel):
            m = model.module
        else:
            m = model
        if hasattr(m, "encoder"):
            enc_params = [p for p in m.encoder.parameters() if p.requires_grad and id(p) not in params_seen]
            if enc_params:
                for p in enc_params:
                    params_seen.add(id(p))
                param_groups.append({"params": enc_params, "lr": args.encoder_lr})
        if hasattr(m, "projector"):
            proj_params = [p for p in m.projector.parameters() if p.requires_grad and id(p) not in params_seen]
            if proj_params:
                for p in proj_params:
                    params_seen.add(id(p))
                param_groups.append({"params": proj_params, "lr": args.projector_lr})
        if hasattr(m, "classifier"):
            cls_params = [p for p in m.classifier.parameters() if p.requires_grad and id(p) not in params_seen]
            if cls_params:
                for p in cls_params:
                    params_seen.add(id(p))
                param_groups.append({"params": cls_params, "lr": args.classifier_lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate_fn, processor=processor, sampling_rate=args.sampling_rate, apply_augmentation=args.use_data_aug, aug=aug_cfg), num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_fn, processor=processor, sampling_rate=args.sampling_rate, apply_augmentation=False, aug=None), num_workers=args.num_workers, pin_memory=True)

        total_steps = len(train_loader) * args.num_epochs // max(1, args.grad_accum_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        criterion = torch.nn.CrossEntropyLoss()
        scaler = amp.GradScaler(enabled=args.use_amp)

        best_val_uar = 0.0
        best_val_metrics = None
        patience_counter = 0

        for epoch in range(args.num_epochs):
            model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []
            optimizer.zero_grad()
            for batch in tqdm(train_loader, desc=f"train {epoch+1}/{args.num_epochs}"):
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                with amp.autocast(enabled=args.use_amp, device_type="cuda" if device.type == "cuda" else "cpu"):
                    outputs = model(input_features)
                    logits = outputs.logits
                    loss = criterion(logits, labels) / max(1, args.grad_accum_steps)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_loss += loss.item() * max(1, args.grad_accum_steps)
                batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                train_preds.extend(batch_preds)
                train_labels.extend(labels.detach().cpu().numpy())

            model.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"eval {epoch+1}/{args.num_epochs}"):
                    input_features = batch["input_features"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_features)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    batch_preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                    val_preds.extend(batch_preds)
                    val_labels.extend(labels.detach().cpu().numpy())

            train_metrics = compute_metrics(train_preds, train_labels)
            val_metrics = compute_metrics(val_preds, val_labels)
            plateau_scheduler.step(val_loss / max(1, len(val_loader)))

            if val_metrics["uar"] > best_val_uar:
                best_val_uar = val_metrics["uar"]
                best_val_metrics = val_metrics.copy()
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                with open(metrics_path, "w") as f:
                    json.dump(val_metrics, f, indent=2)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    break

        fold_results.append(best_val_metrics if best_val_metrics else {"acc": 0, "uar": 0, "war": 0, "f1": 0})

    if fold_results and fold_results[0] is not None:
        avg_metrics = {k: float(np.mean([r[k] for r in fold_results])) for k in fold_results[0].keys()}
        std_metrics = {k: float(np.std([r[k] for r in fold_results])) for k in fold_results[0].keys()}
    else:
        avg_metrics, std_metrics = {}, {}
    with open(os.path.join(args.save_path, "cv_results.json"), "w") as f:
        json.dump({"folds": fold_results, "average_metrics": avg_metrics, "std_metrics": std_metrics}, f, indent=2)

if __name__ == "__main__":
    main()