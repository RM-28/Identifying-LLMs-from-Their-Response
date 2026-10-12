# All models: 5 seeds (0, 1, 2, 3, 4)

Test scores as mean ± std across seeds; [95% CI of the mean, t-distribution].

| model | input | accuracy | macro-F1 | macro-F1 95% CI |
| --- | --- | --- | --- | --- |
| cnnlstm | output | 0.572 ± 0.012 | 0.574 ± 0.015 | [0.555, 0.593] |
| fusion | output | 0.705 ± 0.023 | 0.710 ± 0.019 | [0.686, 0.734] |
| ensemble | output | 0.715 ± 0.023 | 0.716 ± 0.022 | [0.689, 0.744] |
| team_cnn | output | 0.719 ± 0.016 | 0.721 ± 0.014 | [0.704, 0.739] |
| team_rnn | output | 0.607 ± 0.015 | 0.604 ± 0.019 | [0.580, 0.628] |
| cnnlstm | both | 0.564 ± 0.013 | 0.569 ± 0.015 | [0.551, 0.587] |
| fusion | both | 0.691 ± 0.017 | 0.694 ± 0.017 | [0.673, 0.715] |
| ensemble | both | 0.699 ± 0.014 | 0.698 ± 0.015 | [0.679, 0.717] |
| team_cnn | both | 0.712 ± 0.024 | 0.717 ± 0.022 | [0.690, 0.744] |
| team_rnn | both | 0.594 ± 0.023 | 0.597 ± 0.024 | [0.568, 0.627] |
| cnnlstm | input | 0.249 ± 0.008 | 0.247 ± 0.008 | [0.238, 0.257] |
| fusion | input | 0.261 ± 0.009 | 0.252 ± 0.016 | [0.232, 0.271] |
| ensemble | input | 0.254 ± 0.007 | 0.249 ± 0.007 | [0.240, 0.257] |
| team_cnn | input | 0.256 ± 0.009 | 0.252 ± 0.007 | [0.243, 0.261] |
| team_rnn | input | 0.264 ± 0.017 | 0.257 ± 0.019 | [0.233, 0.281] |

Ensemble members on their own (macro-F1):

| ensemble run | cnn | rnn |
| --- | --- | --- |
| ensemble/both | 0.718 ± 0.011 | 0.572 ± 0.052 |
| ensemble/input | 0.247 ± 0.006 | 0.242 ± 0.016 |
| ensemble/output | 0.728 ± 0.024 | 0.621 ± 0.022 |

## Paired comparisons (A − B, macro-F1)

Same seed means same test rows. *Wins* = seeds where A beat B. *McNemar* = seeds where per-row correctness differed at p < 0.05.

| A | B | mean diff ± std | wins | paired t p | McNemar p<0.05 |
| --- | --- | --- | --- | --- | --- |
| cnnlstm/output | fusion/output | -0.136 ± 0.031 | 0/5 | 0.000633 | 5/5 |
| cnnlstm/output | ensemble/output | -0.142 ± 0.029 | 0/5 | 0.000386 | 5/5 |
| cnnlstm/output | team_cnn/output | -0.147 ± 0.021 | 0/5 | 8.95e-05 | 5/5 |
| cnnlstm/output | team_rnn/output | -0.030 ± 0.029 | 1/5 | 0.0791 | 3/5 |
| fusion/output | ensemble/output | -0.006 ± 0.019 | 2/5 | 0.508 | 1/5 |
| fusion/output | team_cnn/output | -0.011 ± 0.018 | 1/5 | 0.223 | 1/5 |
| fusion/output | team_rnn/output | +0.106 ± 0.032 | 5/5 | 0.0018 | 5/5 |
| ensemble/output | team_cnn/output | -0.005 ± 0.009 | 2/5 | 0.263 | 0/5 |
| ensemble/output | team_rnn/output | +0.112 ± 0.035 | 5/5 | 0.00208 | 5/5 |
| team_cnn/output | team_rnn/output | +0.117 ± 0.029 | 5/5 | 0.000843 | 5/5 |
| cnnlstm/both | fusion/both | -0.125 ± 0.022 | 0/5 | 0.000235 | 5/5 |
| cnnlstm/both | ensemble/both | -0.129 ± 0.017 | 0/5 | 6.98e-05 | 5/5 |
| cnnlstm/both | team_cnn/both | -0.148 ± 0.029 | 0/5 | 0.000347 | 5/5 |
| cnnlstm/both | team_rnn/both | -0.028 ± 0.025 | 0/5 | 0.0625 | 1/5 |
| fusion/both | ensemble/both | -0.004 ± 0.007 | 2/5 | 0.275 | 0/5 |
| fusion/both | team_cnn/both | -0.023 ± 0.018 | 1/5 | 0.0444 | 3/5 |
| fusion/both | team_rnn/both | +0.097 ± 0.027 | 5/5 | 0.00134 | 5/5 |
| ensemble/both | team_cnn/both | -0.019 ± 0.021 | 1/5 | 0.111 | 1/5 |
| ensemble/both | team_rnn/both | +0.100 ± 0.025 | 5/5 | 0.00091 | 5/5 |
| team_cnn/both | team_rnn/both | +0.119 ± 0.018 | 5/5 | 0.000126 | 5/5 |
| cnnlstm/input | fusion/input | -0.004 ± 0.017 | 1/5 | 0.592 | 0/5 |
| cnnlstm/input | ensemble/input | -0.002 ± 0.013 | 2/5 | 0.806 | 0/5 |
| cnnlstm/input | team_cnn/input | -0.005 ± 0.007 | 1/5 | 0.229 | 0/5 |
| cnnlstm/input | team_rnn/input | -0.010 ± 0.018 | 2/5 | 0.277 | 0/5 |
| fusion/input | ensemble/input | +0.003 ± 0.014 | 3/5 | 0.663 | 0/5 |
| fusion/input | team_cnn/input | -0.000 ± 0.013 | 2/5 | 0.966 | 0/5 |
| fusion/input | team_rnn/input | -0.006 ± 0.018 | 2/5 | 0.511 | 0/5 |
| ensemble/input | team_cnn/input | -0.003 ± 0.009 | 2/5 | 0.491 | 0/5 |
| ensemble/input | team_rnn/input | -0.009 ± 0.019 | 2/5 | 0.363 | 0/5 |
| team_cnn/input | team_rnn/input | -0.005 ± 0.016 | 3/5 | 0.484 | 0/5 |
| cnnlstm/output | cnnlstm/both | +0.005 ± 0.025 | 2/5 | 0.698 | 1/5 |
| cnnlstm/output | cnnlstm/input | +0.327 ± 0.018 | 5/5 | 2.16e-06 | 5/5 |
| cnnlstm/both | cnnlstm/input | +0.322 ± 0.021 | 5/5 | 4.39e-06 | 5/5 |
| fusion/output | fusion/both | +0.016 ± 0.014 | 4/5 | 0.0685 | 2/5 |
| fusion/output | fusion/input | +0.459 ± 0.018 | 5/5 | 5.12e-07 | 5/5 |
| fusion/both | fusion/input | +0.443 ± 0.023 | 5/5 | 1.71e-06 | 5/5 |
| ensemble/output | ensemble/both | +0.018 ± 0.022 | 4/5 | 0.135 | 2/5 |
| ensemble/output | ensemble/input | +0.468 ± 0.023 | 5/5 | 1.41e-06 | 5/5 |
| ensemble/both | ensemble/input | +0.449 ± 0.012 | 5/5 | 1.33e-07 | 5/5 |
| team_cnn/output | team_cnn/both | +0.005 ± 0.009 | 4/5 | 0.321 | 0/5 |
| team_cnn/output | team_cnn/input | +0.470 ± 0.015 | 5/5 | 2.6e-07 | 5/5 |
| team_cnn/both | team_cnn/input | +0.465 ± 0.023 | 5/5 | 1.51e-06 | 5/5 |
| team_rnn/output | team_rnn/both | +0.007 ± 0.043 | 4/5 | 0.75 | 3/5 |
| team_rnn/output | team_rnn/input | +0.347 ± 0.027 | 5/5 | 9.27e-06 | 5/5 |
| team_rnn/both | team_rnn/input | +0.340 ± 0.031 | 5/5 | 1.7e-05 | 5/5 |
