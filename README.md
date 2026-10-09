# Identifying-LLMs-from-Their-Response

Team Names: Raj Mamidala, Kenny Liang, Aditya Raj

## Team Work distribution
| Member | Model |
| --- | --- |
| Raj Mamidala | RNN |
| Kenny Liang | CNN |
| Aditya Raj | CNN-RNN Hybrid |

### Responsibilities that need to
- **Dataset preparation:** Load and balance the data, then split it into train/val/test sets
- **Model development:** Build and train the CNN, RNN, and CNN-RNN hybrid
- **Training & development:** Use one shared pipeline to see the data, splits and metric on every model
- **Experiments:** Run RQ1, RQ2, RQ3 and RQ4
- **Report:** Document the dataset, models, results, analysis, and lessons learned

### Dataset: 
[LMSYS Chatbot Arena Conversation] https://huggingface.co/datasets/lmsys/chatbot_arena_conversations 


## CNN + RNN combined models

Code: `models/combined_models.py` (models), `train_combined.py` (training + evaluation), `combined_eval.py` (metrics/plots), `run_all.sh`.

| `--model` | What it is |
| --- | --- |
| `cnnlstm` | Embedding → parallel Conv1d (k = 2, 3, 4, 5) → concatenated feature maps → BiLSTM → max + mean pooling → Linear |
| `fusion` | Shared embedding → CNN branch (global max pool) ‖ BiLSTM branch (final hidden states) → concat → Linear |
| `ensemble` | Soft vote: averages the softmax probabilities of the CNN and RNN models |

`--input_mode` picks the text the model sees: `output` (response only), `input` (prompt only), or `both` (`prompt <sep> response`; the response gets priority within `--max_len`).

**Run**

```bash
pip install -r requirements.txt
huggingface-cli login            # the LMSYS dataset is gated: accept its terms on the HF page first
python data/preprocess.py        # writes data/preprocessed/{train,valid,test}.csv

# one run
python train_combined.py --model cnnlstm --input_mode output --epochs 20 --lr 1e-3 --batch_size 64 --max_len 512 --seed 42

# quick smoke test on 100 examples per label
python train_combined.py --model fusion --input_mode both --subset 100 --epochs 2

# everything (3 models x 3 input modes); extra flags go to every run
bash run_all.sh
bash run_all.sh --cnn_ckpt checkpoints/cnn.pt --rnn_ckpt checkpoints/rnn.pt   # ensemble the CNN/RNN checkpoints
```

**Splits.** By default the data is re-split 70/15/15, stratified by label and grouped by prompt (both answers to an arena prompt land in the same split) with `--seed`. Use `--split preprocessed` to keep the splits from `data/preprocess.py` instead.

**Training.** Adam with weight decay (`--weight_decay`), gradient clipping, early stopping on validation macro-F1 (`--patience`). The best checkpoint goes to `checkpoints/combined_<model>_<mode>.pt`.

**Ensemble checkpoints.** `--cnn_ckpt` / `--rnn_ckpt` accept a `torch.save`d `nn.Module`, `{"model": module, "vocab": {token: id}}`, or `{"model_class": "models.cnn:CNN", "model_kwargs": {...}, "state_dict": ..., "vocab": ...}`. The model must take `(ids, lengths)` or just `ids` (padding id 0) and return logits whose classes are in sorted label order. Without these flags, the ensemble trains its own stand-in CNN and BiLSTM.

**Multiple seeds.** `bash run_seeds.sh` repeats the whole sweep for seeds 0–4. Each seed gets a different split and initialisation, and the seeds run in parallel; set `PARALLEL=0` to run them one after another. Results go to `results/seeds/seed<k>/`, and `aggregate_results.py` then writes `results/seeds/summary.{md,json}`. The summary has mean ± std and a 95% CI across seeds, plus paired comparisons between models: on the same seed every model uses the same test rows, and the summary includes a paired t-test across seeds and a McNemar test per seed. Each single run also reports a bootstrap 95% CI (`test.ci95`). cuDNN runs in deterministic mode, so a given seed reproduces exactly.

**Outputs.** `results/combined_<model>_<mode>.json` (accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, training curve, and per-member scores for the ensemble), `results/combined_<model>_<mode>_confusion.png`, and logs in `results/logs/`.
