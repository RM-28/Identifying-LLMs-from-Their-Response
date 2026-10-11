# All models: 5 seeds (0, 1, 2, 3, 4)

Test scores as mean ± std across seeds; [95% CI of the mean, t-distribution].

| model | input | val macro-F1 | test accuracy | test macro-F1 | test macro-F1 95% CI |
| --- | --- | --- | --- | --- | --- |
| cnnlstm | output | 0.556 ± 0.009 | 0.557 ± 0.015 | 0.555 ± 0.018 | [0.532, 0.578] |
| fusion | output | 0.647 ± 0.008 | 0.647 ± 0.012 | 0.649 ± 0.007 | [0.640, 0.657] |
| ensemble | output | 0.655 ± 0.017 | 0.659 ± 0.027 | 0.658 ± 0.028 | [0.623, 0.692] |
| team_cnn | output | 0.671 ± 0.007 | 0.680 ± 0.005 | 0.681 ± 0.006 | [0.674, 0.688] |
| team_rnn | output | 0.590 ± 0.008 | 0.576 ± 0.004 | 0.576 ± 0.005 | [0.571, 0.582] |

Best by mean validation macro-F1: **team_cnn/output** (val 0.671 ± 0.007, test 0.681 ± 0.006)

Ensemble members on their own (macro-F1):

| ensemble run | cnn | rnn |
| --- | --- | --- |
| ensemble/output | 0.681 ± 0.006 | 0.584 ± 0.013 |

## Paired comparisons (A − B, macro-F1)

Same seed means same test rows. *Wins* = seeds where A beat B. *McNemar* = seeds where per-row correctness differed at p < 0.05.

| A | B | mean diff ± std | wins | paired t p | McNemar p<0.05 |
| --- | --- | --- | --- | --- | --- |
| cnnlstm/output | fusion/output | -0.094 ± 0.025 | 0/5 | 0.00105 | 5/5 |
| cnnlstm/output | ensemble/output | -0.103 ± 0.022 | 0/5 | 0.000469 | 5/5 |
| cnnlstm/output | team_cnn/output | -0.126 ± 0.020 | 0/5 | 0.000155 | 5/5 |
| cnnlstm/output | team_rnn/output | -0.022 ± 0.021 | 0/5 | 0.0821 | 1/5 |
| fusion/output | ensemble/output | -0.009 ± 0.033 | 1/5 | 0.573 | 4/5 |
| fusion/output | team_cnn/output | -0.032 ± 0.007 | 0/5 | 0.000661 | 4/5 |
| fusion/output | team_rnn/output | +0.072 ± 0.006 | 5/5 | 1.29e-05 | 5/5 |
| ensemble/output | team_cnn/output | -0.023 ± 0.027 | 1/5 | 0.132 | 1/5 |
| ensemble/output | team_rnn/output | +0.081 ± 0.031 | 5/5 | 0.00428 | 5/5 |
| team_cnn/output | team_rnn/output | +0.104 ± 0.006 | 5/5 | 3.48e-06 | 5/5 |
