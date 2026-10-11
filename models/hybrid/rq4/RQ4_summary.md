# RQ4: What characteristics distinguish the LLMs?

Data: the team's preprocessed CSVs, response text only, 4 LLMs (claude-v1, gpt-3.5-turbo, palm-2, vicuna-13b).
Stats use the training split (6,481 responses, about 1,610 per LLM). Ablation models are trained with the
unchanged pipeline in `models/hybrid/train_combined.py`.

## 1. Per-LLM statistics (training data)

Full table: `results/llm_stats.csv`. Plots: `results/length_tokens.png`, `results/ttr.png`, `results/formatting.png`.
Values are means per response unless marked "%" (= % of responses with at least one).

| | claude-v1 | gpt-3.5-turbo | palm-2 | vicuna-13b |
| --- | --- | --- | --- | --- |
| characters | 1091 | 679 | 1024 | 868 |
| tokens | 214 | 136 | 214 | 174 |
| type-token ratio | 0.553 | 0.616 | 0.494 | 0.543 |
| vocabulary size (lowercased) | 23,642 | 17,546 | 18,188 | 18,463 |
| newlines | 14.7 | 7.8 | 11.8 | 8.5 |
| % with bold (`**...**`) | 0.0 | 0.2 | 12.5 | 0.2 |
| % with bullets | 36.6 | 3.9 | 26.2 | 5.7 |
| % with numbered list | 26.6 | 19.4 | 9.9 | 21.2 |
| % with headers | 2.6 | 2.1 | 2.7 | 3.1 |
| % with code block | 8.6 | 8.8 | 10.3 | 8.5 |
| % with "As an AI" | 0.1 | 7.4 | 0.0 | 3.5 |
| % with "I apologize" | 7.3 | 0.2 | 0.1 | 0.3 |
| % with "Certainly" | 0.2 | 1.2 | 0.3 | 1.3 |

What the numbers show:
- **Length:** gpt-3.5-turbo responses are the shortest (136 tokens on average); claude-v1 and palm-2 are the longest (214).
- **Lexical diversity:** gpt-3.5-turbo has the highest type-token ratio (0.616) and palm-2 the lowest (0.494). TTR falls as texts get longer, so part of this difference follows from length.
- **Vocabulary:** claude-v1 uses the most different tokens (23,642; the others are 17.5k–18.5k).
- **Formatting:** the clearest single markers are bold, used almost only by palm-2 (12.5% vs ≤0.2%), and bullets, used by claude-v1 (36.6%) and palm-2 (26.2%) far more than by gpt-3.5-turbo (3.9%) and vicuna-13b (5.7%). Headers and code blocks are about equal across LLMs.
- **Phrases:** "As an AI" appears mostly in gpt-3.5-turbo (7.4%) and vicuna-13b (3.5%); "I apologize" mostly in claude-v1 (7.3%). "Certainly" is rare for all (≤1.3%).

## 2. Ablation (test set, mean ± std over seeds 0, 1, 2)

Full table: `results/ablation/ablation_results.md` / `.csv`. Drop = baseline − setting (positive = worse).

| model | setting | test accuracy | test macro-F1 | acc drop | macro-F1 drop |
| --- | --- | --- | --- | --- | --- |
| CNN | baseline | 0.682 ± 0.005 | 0.683 ± 0.006 | – | – |
| CNN | markdown stripped | 0.661 ± 0.008 | 0.663 ± 0.007 | +0.021 | +0.020 |
| CNN | truncated to 50 tokens | 0.603 ± 0.010 | 0.600 ± 0.016 | +0.079 | +0.083 |
| CNN | whitespace removed | 0.685 ± 0.002 | 0.684 ± 0.007 | −0.003 | −0.001 |
| Parallel | baseline | 0.643 ± 0.011 | 0.646 ± 0.005 | – | – |
| Parallel | markdown stripped | 0.629 ± 0.009 | 0.630 ± 0.006 | +0.014 | +0.016 |
| Parallel | truncated to 50 tokens | 0.584 ± 0.005 | 0.581 ± 0.008 | +0.059 | +0.065 |
| Parallel | whitespace removed | 0.651 ± 0.007 | 0.652 ± 0.013 | −0.008 | −0.005 |

What the numbers show:
- **Truncating to 50 tokens hurts most:** macro-F1 drops by 0.083 (CNN) and 0.065 (Parallel), and it is lower than baseline on every seed. Both models still score well above chance (0.25 for 4 balanced classes) with only the first 50 tokens.
- **Stripping markdown hurts a little:** −0.020 (CNN) and −0.016 (Parallel), lower than baseline on every seed for both models.
- **Removing newlines/whitespace makes no measurable difference:** the change (+0.001, +0.005) is within the seed-to-seed spread, and per-seed results go both ways. The pipeline tokenizer (`word_tokenize`) already drops whitespace, so the models never saw newlines in the baseline either. This setting therefore cannot test whether newlines carry signal for these models.
- The CNN is above the Parallel model in every setting.

## 3. Which signals matter most

- **Structural:** the formatting statistics differ strongly between LLMs (bold for palm-2, bullets for claude-v1 and palm-2), and removing markdown symbols costs the models about 0.02 macro-F1. Newline counts also differ (7.8 to 14.7 per response), but the models cannot use them because of the tokenizer.
- **Length:** lengths differ between LLMs, and cutting the text to 50 tokens gives the largest drop (0.065–0.083). This removes both the length information and most of the content, so the ablation does not separate the two.
- **Linguistic:** with markdown stripped the CNN still reaches 0.663 macro-F1 (baseline 0.683), so most of the performance remains without the formatting symbols. (The two settings were run separately, not combined.) The phrase statistics ("As an AI", "I apologize") show word-level markers, but each appears in at most 7.4% of one LLM's responses.

In short: removing formatting symbols costs a small but consistent ~0.02 macro-F1, and most of the performance remains without them. Reducing the amount of text to 50 tokens costs more (0.065–0.083). The whitespace setting is uninformative with this tokenizer.
