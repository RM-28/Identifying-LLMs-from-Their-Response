# Identifying-LLMs-from-Their-Response

Team Names: Raj Mamidala, Kenny Liang, Aditya Raj

## Team Work distribution
| Member | Model |
| --- | --- |
| Raj Mamidala | RNN |
| Kenny Liang | CNN |
| Aditya Raj | CNN-RNN Hybrid |

### Responsibilities needed for the project
- **Dataset preparation:** Load and balance the data, then split it into train/val/test sets
- **Model development:** Build and train the CNN, RNN, and CNN-RNN hybrid
- **Training & development:** Use one shared pipeline to see the data, splits and metric on every model
- **Experiments:** Run RQ1, RQ2, RQ3 and RQ4
- **Report:** Document the dataset, models, results, analysis, and lessons learned

### Dataset: 
[LMSYS Chatbot Arena Conversation] https://huggingface.co/datasets/lmsys/chatbot_arena_conversations 


## CNN + RNN combined models

All code is in `models/hybrid/`: `team_models.py` (models), `combined_models.py` (soft-vote ensemble + model list), `train_combined.py` (training + evaluation), `combined_eval.py` (metrics/plots), `run_all.sh`, `run_seeds.sh`, `aggregate_results.py`. Run everything from the repo root.

Every model is built from the teammates' code, imported unchanged: Kenny's CNN (`models/cnn/model/conv.py`, `CNN_NLP`) and Matteo's RNN (`models/rnn/rnn.py`, `RNN`), both starting from the fastText embeddings (loaded with `models/cnn/load_embeddings.py`).

| `--model` | What it is |
| --- | --- |
| `team_cnn`, `team_rnn` | Kenny's CNN and Matteo's RNN on their own (baselines) |
| `cnnlstm` | Stacked: Kenny's embedding + convs (k = 3, 4, 5 × 100) → Matteo's RNN (unidirectional LSTM, 128) reading the conv feature maps → final hidden state → Linear |
| `fusion` | Parallel: Kenny's CNN ‖ Matteo's RNN sharing one embedding layer, heads removed → concat (300 + 128) → Linear |
| `ensemble` | Soft vote: Kenny's CNN and Matteo's RNN trained separately, softmax probabilities averaged |

The task uses the response text only (`--input_mode output`).

**Run**

```bash
pip install -r requirements.txt
python data/preprocess.py                 # writes data/preprocessed/{train,valid,test}.csv
python models/cnn/embedding_download.py   # fastText vectors into fastText/ (not committed)

# one run
python models/hybrid/train_combined.py --model cnnlstm --seed 0

# quick smoke test on 100 examples per label
python models/hybrid/train_combined.py --model fusion --subset 100 --epochs 2

# all 5 models for one seed; MODELS="cnnlstm fusion" picks a subset
bash models/hybrid/run_all.sh

# all 5 models x seeds 0-4, then the summary table
bash models/hybrid/run_seeds.sh
```

**Data.** The team's train/valid/test CSVs from `data/preprocess.py` (`--split preprocessed`, the default), tokenized with nltk `word_tokenize`, cut to 256 tokens, vocab = train tokens seen at least twice.

**Training.** Same loop for every model, following `models/rnn/train.py`: Adam (lr 1e-3), batch 32, at most 12 epochs, early stopping on validation macro-F1 (patience 3); plus gradient clipping. The best checkpoint goes to `models/hybrid/checkpoints/` (not committed).

**Multiple seeds.** `run_seeds.sh` runs seeds 0–4; the split stays the same, the seed changes initialisation and batch order. `aggregate_results.py` writes `models/hybrid/results/seeds/summary.{md,json}` with mean ± std across seeds, the best model by validation macro-F1, and paired comparisons (paired t-test across seeds, McNemar per seed).

**Outputs.** `models/hybrid/results/seeds/seed<k>/combined_<model>_output.json` (accuracy, macro-F1, per-class scores, confusion matrix, training curve), a confusion-matrix PNG, and logs in `logs/`.
