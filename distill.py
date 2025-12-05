import os
import re
import math
import glob
import json
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.amp as amp
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from transformers import AutoModelForAudioClassification, AutoProcessor
from torchaudio.transforms import FrequencyMasking, TimeMasking
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="./outputs/distill_loso")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--teacher_model", type=str, default="openai/whisper-large-v2")
    p.add_argument("--student_model", type=str, default="openai/whisper-small")
    p.add_argument("--teacher_path", type=str, nargs="*")
    p.add_argument("--teacher_dir", type=str)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--student_last_n_layer", type=int, default=4)
    p.add_argument("--use_augmentation", action="store_true")
    p.add_argument("--freq_mask_param", type=int, default=27)
    p.add_argument("--time_mask_param", type=int, default=40)
    p.add_argument("--freq_mask_prob", type=float, default=0.7)
    p.add_argument("--time_mask_prob", type=float, default=0.7)
    p.add_argument("--amplitude_scale_prob", type=float, default=0.3)
    p.add_argument("--amplitude_scale_min", type=float, default=0.8)
    p.add_argument("--amplitude_scale_max", type=float, default=1.2)
    p.add_argument("--use_ce", action="store_true", default=True)
    p.add_argument("--use_kl", action="store_true", default=True)
    p.add_argument("--use_cka", action="store_true", default=True)
    p.add_argument("--ce_weight", type=float, default=1.0)
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--cka_weight", type=float, default=1.0)
    p.add_argument("--adaptive_temp", action="store_true")
    p.add_argument("--temp_method", type=str, default="entropy", choices=["confidence", "entropy"])
    p.add_argument("--initial_temp", type=float, default=4.0)
    p.add_argument("--min_temp", type=float, default=2.0)
    p.add_argument("--max_temp", type=float, default=10.0)
    p.add_argument("--temp_decay", action="store_true")
    p.add_argument("--fold_idx", type=int, default=-1)
    p.add_argument("--sample_ratio", type=float, default=1.0)
    return p.parse_args()

def get_config(a):
    return {
        "device": a.device,
        "save_path": a.save_path,
        "num_epochs": a.num_epochs,
        "patience": 10,
        "use_amp": True,
        "seed": a.seed,
        "data": {
            "path": a.data_path,
            "batch_size": a.batch_size,
            "num_workers": 8,
            "use_augmentation": a.use_augmentation,
            "sampling_rate": 16000,
            "sample_ratio": a.sample_ratio,
            "augmentation": {
                "freq_mask_param": a.freq_mask_param,
                "time_mask_param": a.time_mask_param,
                "freq_mask_prob": a.freq_mask_prob,
                "time_mask_prob": a.time_mask_prob,
                "amplitude_scale_prob": a.amplitude_scale_prob,
                "amplitude_scale_range": (a.amplitude_scale_min, a.amplitude_scale_max)
            }
        },
        "model": {
            "num_classes": 4,
            "teacher": {"id": a.teacher_model, "path_list": []},
            "student": {"id": a.student_model, "layerdrop": 0.0, "dropout": 0.0, "last_n_layers": a.student_last_n_layer, "activation_dropout": 0.0, "attention_dropout": 0.0}
        },
        "optimizer": {
            "group_config": {
                "classifier": {"lr": 5e-4, "wd": 0.05},
                "projector": {"lr": 2e-4, "wd": 0.01},
                "encoder.layers": {"lr": 1e-4, "wd": 0.005},
                "encoder.layer_norm": {"lr": 1e-5, "wd": 0.0}
            },
            "grad_accum_steps": a.grad_accum_steps,
            "warmup_epochs": 2,
            "grad_clip": 0.5
        },
        "distill": {
            "ce": {"use": a.use_ce, "weight": a.ce_weight},
            "kl": {"use": a.use_kl, "weight": a.kl_weight, "temperature": {"adaptive": a.adaptive_temp, "initial": a.initial_temp, "min": a.min_temp, "max": a.max_temp, "decay": a.temp_decay, "method": a.temp_method}},
            "cka": {"use": a.use_cka, "weight": a.cka_weight, "feature_matching": {"student_layers": [-1], "teacher_layers": [-1], "match_weights": [1.0]}}
        }
    }

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

def adaptive_temperature(teacher_logits, student_logits, min_temp, max_temp, epsilon=1e-6, max_ratio=3.0, ema_factor=0.9, prev_temp=None):
    with torch.no_grad():
        tp = torch.softmax(teacher_logits, dim=-1)
        sp = torch.softmax(student_logits, dim=-1)
        tc = tp.max(dim=-1)[0].mean()
        sc = sp.max(dim=-1)[0].mean()
        conf_ratio = (tc / (sc + epsilon)).clamp(1.0, max_ratio)
        norm_ratio = (conf_ratio - 1.0) / (max_ratio - 1.0)
        current_temp = min_temp + norm_ratio * (max_temp - min_temp)
        if prev_temp is not None:
            temp = ema_factor * prev_temp + (1 - ema_factor) * current_temp
        else:
            temp = current_temp
        return temp.item()

def entropy_based_temperature(teacher_logits, student_logits, min_temp, max_temp, epsilon=1e-6):
    with torch.no_grad():
        tp = torch.softmax(teacher_logits, dim=-1)
        sp = torch.softmax(student_logits, dim=-1)
        te = -torch.sum(tp * torch.log(tp + epsilon), dim=-1).mean()
        se = -torch.sum(sp * torch.log(sp + epsilon), dim=-1).mean()
        ratio = (se / (te + epsilon)).clamp(1.0, 3.0)
        norm = (ratio - 1.0) / 2.0
        temp = min_temp + norm * (max_temp - min_temp)
        return temp.item()

def collate_fn(batch):
    return {"audio_arrays": [x["audio"]["array"] for x in batch], "labels": torch.tensor([x["label"] for x in batch])}

def get_cosine_temp(epoch, total_epochs, initial_temp, min_temp):
    return min_temp + 0.5 * (initial_temp - min_temp) * (1 + math.cos(epoch / total_epochs * math.pi))

def compute_metrics(preds, labels):
    return {"acc": accuracy_score(labels, preds), "uar": recall_score(labels, preds, average="macro"), "war": recall_score(labels, preds, average="weighted"), "f1": f1_score(labels, preds, average="macro")}

class CKALoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, SH, TH):
        dT = TH.size(-1)
        dS = SH.size(-1)
        SH = SH.contiguous().view(-1, dS).to(SH.device, torch.float64)
        TH = TH.contiguous().view(-1, dT).to(TH.device, torch.float64)
        SH = SH - SH.mean(0, keepdim=True)
        TH = TH - TH.mean(0, keepdim=True)
        num = torch.norm(SH.t().matmul(TH), "fro")
        den1 = torch.norm(SH.t().matmul(SH), "fro") + self.eps
        den2 = torch.norm(TH.t().matmul(TH), "fro") + self.eps
        cka = 1 - num / torch.sqrt(den1 * den2)
        return cka

def calculate_loss(student_logits, teacher_logits, labels, s_hidden, t_hidden, config, temp, weights):
    use_ce = config["distill"]["ce"]["use"]
    use_kl = config["distill"]["kl"]["use"]
    use_cka = config["distill"]["cka"]["use"]
    loss_ce = torch.tensor(0.0, device=student_logits.device)
    if use_ce:
        loss_ce = nn.CrossEntropyLoss()(student_logits, labels)
    loss_kl = torch.tensor(0.0, device=student_logits.device)
    if use_kl:
        loss_kl = nn.KLDivLoss(reduction="batchmean")(torch.log_softmax(student_logits / temp, dim=-1), torch.softmax(teacher_logits / temp, dim=-1)) * (temp ** 2)
    loss_cka = torch.tensor(0.0, device=student_logits.device)
    if use_cka:
        cka = CKALoss()
        for s_idx, t_idx, cka_weight in zip(config["distill"]["cka"]["feature_matching"]["student_layers"], config["distill"]["cka"]["feature_matching"]["teacher_layers"], config["distill"]["cka"]["feature_matching"]["match_weights"]):
            loss_cka += cka(s_hidden[s_idx], t_hidden[t_idx]) * cka_weight
    total_loss = weights["ce"] * loss_ce + weights["kl"] * loss_kl + weights["cka"] * loss_cka
    return total_loss, loss_ce, loss_kl, loss_cka

def get_trainable_patterns(model_config, last_n_layers=3):
    total_layers = model_config.encoder_layers
    trainable_layers = "|".join(str(total_layers + i) for i in range(-last_n_layers, 0))
    return {"classifier": r"^classifier.*", "projector": r"^projector.*", "encoder.layers": f"^encoder\\.layers\\.({trainable_layers}).*", "encoder.layer_norm": r"^encoder\.layer_norm.*"}

def apply_smart_augmentation(input_features, attention_mask=None, config=None):
    if not config or not config["data"]["use_augmentation"]:
        return input_features
    device = input_features.device
    b, n_mels, t = input_features.shape
    if attention_mask is not None:
        valid_lengths = attention_mask.sum(dim=1)
    else:
        valid_lengths = torch.full((b,), int(t * 0.8), device=device)
    out = input_features.clone()
    for i in range(b):
        if random.random() < 0.5:
            x = out[i]
            if random.random() < config["data"]["augmentation"]["freq_mask_prob"]:
                x = FrequencyMasking(freq_mask_param=config["data"]["augmentation"]["freq_mask_param"]).to(device)(x.unsqueeze(0)).squeeze(0)
            if random.random() < config["data"]["augmentation"]["time_mask_prob"]:
                max_time_mask = min(config["data"]["augmentation"]["time_mask_param"], max(1, valid_lengths[i].item() // 4))
                if max_time_mask > 0:
                    x = TimeMasking(time_mask_param=max_time_mask).to(device)(x.unsqueeze(0)).squeeze(0)
            if random.random() < config["data"]["augmentation"]["amplitude_scale_prob"]:
                scale_min, scale_max = config["data"]["augmentation"]["amplitude_scale_range"]
                x = x * random.uniform(scale_min, scale_max)
            out[i] = x
    return out

def prepare_speaker_independent_data(config):
    dataset = load_any_dataset(config["data"]["path"])
    all_data = []
    groups = []
    for session in dataset.values() if isinstance(dataset, dict) else [dataset]:
        for item in session:
            spk = os.path.basename(item["audio"]["path"]).split("_")[0]
            all_data.append(item)
            groups.append(spk)
    all_indices = np.arange(len(all_data))
    return all_data, all_indices, groups

def train_fold(config, fold_idx, train_indices, val_indices, all_data, teacher_path):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    teacher_proc = AutoProcessor.from_pretrained(config["model"]["teacher"]["id"])
    student_proc = AutoProcessor.from_pretrained(config["model"]["student"]["id"])
    teacher = AutoModelForAudioClassification.from_pretrained(config["model"]["teacher"]["id"], num_labels=config["model"]["num_classes"]).to(device)
    if teacher_path and os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()
    student = AutoModelForAudioClassification.from_pretrained(config["model"]["student"]["id"], num_labels=config["model"]["num_classes"], encoder_layerdrop=config["model"]["student"]["layerdrop"], dropout=config["model"]["student"]["dropout"], output_hidden_states=True).to(device)
    trainable_patterns = get_trainable_patterns(student.config, last_n_layers=config["model"]["student"]["last_n_layers"])
    for name, param in student.named_parameters():
        param.requires_grad = any(re.fullmatch(pattern, name) for pattern in trainable_patterns.values())
    optimizer_groups = []
    used = set()
    for group_name, pattern in trainable_patterns.items():
        grp = []
        for name, p in student.named_parameters():
            if p.requires_grad and name not in used and re.fullmatch(pattern, name):
                grp.append(p)
                used.add(name)
        if grp:
            cfg = config["optimizer"]["group_config"].get(group_name, {})
            optimizer_groups.append({"params": grp, "lr": cfg.get("lr", 1e-3), "initial_lr": cfg.get("lr", 1e-3), "weight_decay": cfg.get("wd", 0.0)})
    optimizer = AdamW(optimizer_groups, betas=(0.9, 0.98))
    train_data = [all_data[i] for i in train_indices]
    val_data = [all_data[i] for i in val_indices]
    if config["data"].get("sample_ratio", 1.0) < 1.0:
        sample_size = max(1, int(len(train_data) * config["data"]["sample_ratio"]))
        train_data = random.sample(train_data, sample_size)
    train_loader = DataLoader(train_data, batch_size=config["data"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=config["data"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config["data"]["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=config["data"]["num_workers"], pin_memory=True)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    use_ce = config["distill"]["ce"]["use"]
    use_kl = config["distill"]["kl"]["use"]
    use_cka = config["distill"]["cka"]["use"]
    weights = {"ce": config["distill"]["ce"]["weight"] if use_ce else 0.0, "kl": config["distill"]["kl"]["weight"] if use_kl else 0.0, "cka": config["distill"]["cka"]["weight"] if use_cka else 0.0}
    scaler = amp.GradScaler(enabled=config.get("use_amp", True))
    best_uar = 0.0
    best_metrics = None
    best_epoch = 0
    patience_counter = 0
    prev_temp = None
    for epoch in range(config["num_epochs"]):
        if config["distill"]["kl"]["temperature"].get("decay", False):
            current_temp = get_cosine_temp(epoch, config["num_epochs"], config["distill"]["kl"]["temperature"]["initial"], config["distill"]["kl"]["temperature"]["min"])
        else:
            current_temp = config["distill"]["kl"]["temperature"]["initial"]
        student.train()
        train_loss = 0.0
        train_celoss = 0.0
        train_klloss = 0.0
        train_ckaloss = 0.0
        all_spreds = []
        all_tpreds = []
        all_labels = []
        with tqdm(train_loader, desc=f"train {epoch+1}/{config['num_epochs']}") as pbar:
            optimizer.zero_grad()
            for step, batch in enumerate(pbar):
                current_step = epoch * len(train_loader) + step
                warmup_steps = config["optimizer"]["warmup_epochs"] * len(train_loader)
                if current_step < warmup_steps:
                    scale = min(1.0, current_step / max(1, warmup_steps))
                    for pg in optimizer.param_groups:
                        pg["lr"] = pg["initial_lr"] * scale
                t_inputs = teacher_proc(batch["audio_arrays"], sampling_rate=config["data"]["sampling_rate"], return_tensors="pt", return_attention_mask=True).to(device)
                s_inputs = student_proc(batch["audio_arrays"], sampling_rate=config["data"]["sampling_rate"], return_tensors="pt", return_attention_mask=True).to(device)
                if config["data"]["use_augmentation"]:
                    s_inputs["input_features"] = apply_smart_augmentation(s_inputs["input_features"], attention_mask=s_inputs.get("attention_mask"), config=config)
                t_features = t_inputs["input_features"]
                s_features = s_inputs["input_features"]
                labels = batch["labels"].to(device)
                with amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=config.get("use_amp", True)):
                    with torch.no_grad():
                        t_out = teacher(t_features, output_hidden_states=True)
                        t_logits = t_out.logits
                    s_out = student(s_features, output_hidden_states=True)
                    s_logits = s_out.logits
                    if not config["distill"]["kl"]["temperature"]["adaptive"]:
                        if config["distill"]["kl"]["temperature"].get("decay", False):
                            current_temp = get_cosine_temp(epoch, config["num_epochs"], config["distill"]["kl"]["temperature"]["initial"], config["distill"]["kl"]["temperature"]["min"])
                        else:
                            current_temp = config["distill"]["kl"]["temperature"]["initial"]
                        batch_temp = current_temp
                    else:
                        if config["distill"]["kl"]["temperature"]["method"] == "confidence":
                            batch_temp = adaptive_temperature(t_out.logits, s_out.logits, config["distill"]["kl"]["temperature"]["min"], config["distill"]["kl"]["temperature"]["max"])
                        else:
                            batch_temp = entropy_based_temperature(t_out.logits, s_out.logits, config["distill"]["kl"]["temperature"]["min"], config["distill"]["kl"]["temperature"]["max"])
                        prev_temp = batch_temp
                    s_hidden = s_out.hidden_states
                    t_hidden = t_out.hidden_states
                    total_loss, loss_ce, loss_kl, loss_cka = calculate_loss(s_logits, t_logits, labels, s_hidden, t_hidden, config, batch_temp, weights)
                    total_loss = total_loss / max(1, config["optimizer"]["grad_accum_steps"])
                scaler.scale(total_loss).backward()
                if (step + 1) % config["optimizer"]["grad_accum_steps"] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), config["optimizer"]["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                train_loss += total_loss.item() * max(1, config["optimizer"]["grad_accum_steps"])
                train_celoss += loss_ce.item()
                train_klloss += loss_kl.item()
                train_ckaloss += loss_cka.item()
                sp = s_logits.argmax(dim=-1)
                tp = t_logits.argmax(dim=-1)
                all_spreds.extend(sp.detach().cpu().numpy())
                all_tpreds.extend(tp.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                pbar.set_postfix({"loss": total_loss.item() * max(1, config["optimizer"]["grad_accum_steps"]), "acc": accuracy_score(all_labels, all_spreds) if all_labels else 0})
        student.eval()
        val_loss = 0.0
        val_celoss = 0.0
        val_klloss = 0.0
        val_ckaloss = 0.0
        all_val_spreds = []
        all_val_labels = []
        with torch.no_grad():
            with tqdm(val_loader, desc=f"eval {epoch+1}/{config['num_epochs']}") as pbar:
                for batch in pbar:
                    t_inputs = teacher_proc(batch["audio_arrays"], sampling_rate=config["data"]["sampling_rate"], return_tensors="pt").input_features.to(device)
                    s_inputs = student_proc(batch["audio_arrays"], sampling_rate=config["data"]["sampling_rate"], return_tensors="pt").input_features.to(device)
                    labels = batch["labels"].to(device)
                    with amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=config.get("use_amp", True)):
                        t_out = teacher(t_inputs, output_hidden_states=True)
                        t_logits = t_out.logits
                        s_out = student(s_inputs, output_hidden_states=True)
                        s_logits = s_out.logits
                        s_hidden = s_out.hidden_states
                        t_hidden = t_out.hidden_states
                        total_loss, loss_ce, loss_kl, loss_cka = calculate_loss(s_logits, t_logits, labels, s_hidden, t_hidden, config, current_temp, weights)
                    val_loss += total_loss.item()
                    val_celoss += loss_ce.item()
                    val_klloss += loss_kl.item()
                    val_ckaloss += loss_cka.item()
                    sp = s_logits.argmax(dim=-1)
                    all_val_labels.extend(labels.detach().cpu().numpy())
                    all_val_spreds.extend(sp.detach().cpu().numpy())
        train_metrics = compute_metrics(all_spreds, all_labels)
        val_metrics = compute_metrics(all_val_spreds, all_val_labels)
        scheduler.step(val_metrics["uar"])
        if val_metrics["uar"] > best_uar:
            best_uar = val_metrics["uar"]
            best_metrics = val_metrics
            best_epoch = epoch + 1
            patience_counter = 0
            model_save_path = os.path.join(config["save_path"], f"fold{fold_idx+1}_uar_{best_uar:.4f}.pth")
            torch.save(student.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                break
    student.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = student_proc(batch["audio_arrays"], sampling_rate=config["data"]["sampling_rate"], return_tensors="pt").input_features.to(device)
            labels = batch["labels"].to(device)
            outputs = student(inputs)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_true.extend(labels.detach().cpu().numpy())
    cm = confusion_matrix(all_true, all_preds)
    return {"fold": fold_idx + 1, "best_epoch": best_epoch, "metrics": best_metrics, "confusion_matrix": cm.tolist(), "ablation": {"ce": use_ce, "kl": use_kl, "cka": use_cka, "weights": weights}}

def resolve_teacher_paths(teacher_dir, teacher_paths_arg, actual_folds):
    if teacher_paths_arg and len(teacher_paths_arg) > 0:
        return teacher_paths_arg
    if teacher_dir and os.path.isdir(teacher_dir):
        files = glob.glob(os.path.join(teacher_dir, "fold*_best_model.pt")) + glob.glob(os.path.join(teacher_dir, "fold*_uar_*.pth"))
        pairs = []
        for f in files:
            m = re.search(r"fold(\d+)", os.path.basename(f))
            if m:
                pairs.append((int(m.group(1)), f))
        pairs.sort(key=lambda x: x[0])
        out = []
        for i in range(1, actual_folds + 1):
            found = [p for p in pairs if p[0] == i]
            out.append(found[0][1] if found else None)
        return out
    return [None] * actual_folds

def main():
    args = parse_args()
    config = get_config(args)
    set_seed(config["seed"])
    os.makedirs(config["save_path"], exist_ok=True)
    all_data, all_indices, groups = prepare_speaker_independent_data(config)
    logo = LeaveOneGroupOut()
    splits = list(logo.split(all_indices, groups=groups))
    actual_folds = len(splits)
    config["model"]["teacher"]["path_list"] = resolve_teacher_paths(args.teacher_dir, args.teacher_path, actual_folds)
    if args.fold_idx >= 0 and args.fold_idx < actual_folds:
        fold_indices = [args.fold_idx]
    else:
        fold_indices = list(range(actual_folds))
    results = []
    for fi in fold_indices:
        train_indices, val_indices = splits[fi]
        tp = config["model"]["teacher"]["path_list"][fi] if fi < len(config["model"]["teacher"]["path_list"]) else None
        r = train_fold(config, fi, train_indices, val_indices, all_data, tp)
        results.append(r)
    if len(results) > 1:
        avg_metrics = {"acc": float(np.mean([r["metrics"]["acc"] for r in results])), "uar": float(np.mean([r["metrics"]["uar"] for r in results])), "war": float(np.mean([r["metrics"]["war"] for r in results])), "f1": float(np.mean([r["metrics"]["f1"] for r in results]))}
        std_metrics = {"acc": float(np.std([r["metrics"]["acc"] for r in results])), "uar": float(np.std([r["metrics"]["uar"] for r in results])), "war": float(np.std([r["metrics"]["war"] for r in results])), "f1": float(np.std([r["metrics"]["f1"] for r in results]))}
        total_cm = np.zeros((config["model"]["num_classes"], config["model"]["num_classes"]))
        for r in results:
            total_cm += np.array(r["confusion_matrix"])
    else:
        avg_metrics = results[0]["metrics"]
        std_metrics = None
        total_cm = results[0]["confusion_matrix"]
    with open(os.path.join(config["save_path"], "results.json"), "w") as f:
        json.dump({"config": config, "fold_results": results, "avg_metrics": avg_metrics, "std_metrics": std_metrics, "total_confusion_matrix": total_cm.tolist() if isinstance(total_cm, np.ndarray) else total_cm}, f, indent=2)

if __name__ == "__main__":
    main()