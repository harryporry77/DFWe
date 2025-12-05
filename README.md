# DFWe: LOSO Fine-tuning and Distillation

This directory contains two scripts:
- `finetune.py`: Fine-tune audio classification models in a speaker-independent (LOSO) setup
- `distill.py`: Distill from a teacher model to a student (supports CE, KL, CKA)

## Installation
```
pip install torch torchaudio transformers datasets scikit-learn tqdm
```
Recommended: `transformers>=4.42`, `torch>=2.0`.

## Data
- `--data_path` points to data loadable by `datasets`: via `load_dataset(path)` or `load_from_disk(path)`
- Required fields: `audio.array`, `audio.path` (for speaker grouping), `label`

## Usage
- Fine-tuning:
```
python code/finetune.py --data_path /path/to/iemocap --model_id openai/whisper-tiny --save_path ./outputs/finetune_loso --use_data_aug --unfreeze_last_n_encoder_layers 4
```
- Distillation:
```
python code/distill.py --data_path /path/to/iemocap --teacher_model openai/whisper-large-v2 --student_model openai/whisper-small --save_path ./outputs/distill_loso --teacher_dir ./outputs/finetune_loso --use_augmentation --adaptive_temp --student_last_n_layer 4
```

## Outputs
- Fine-tuning: `fold{N}_best_model.pt`, `fold{N}_best_metrics.json`, `cv_results.json`
- Distillation: `fold{N}_uar_*.pth`, `results.json`




