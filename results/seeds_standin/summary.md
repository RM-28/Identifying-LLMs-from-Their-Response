# All models: 5 seeds (0, 1, 2, 3, 4)

Test scores as mean ± std across seeds; [95% CI of the mean, t-distribution].

| model | input | accuracy | macro-F1 | macro-F1 95% CI |
| --- | --- | --- | --- | --- |
| cnnlstm | output | 0.725 ± 0.016 | 0.728 ± 0.014 | [0.710, 0.746] |
| fusion | output | 0.680 ± 0.009 | 0.679 ± 0.010 | [0.667, 0.692] |
| ensemble | output | 0.689 ± 0.036 | 0.688 ± 0.038 | [0.641, 0.736] |
| team_cnn | output | 0.719 ± 0.016 | 0.721 ± 0.014 | [0.704, 0.739] |
| team_rnn | output | 0.607 ± 0.015 | 0.604 ± 0.019 | [0.580, 0.628] |
| cnnlstm | both | 0.698 ± 0.019 | 0.700 ± 0.021 | [0.674, 0.726] |
| fusion | both | 0.648 ± 0.012 | 0.651 ± 0.014 | [0.635, 0.668] |
| ensemble | both | 0.650 ± 0.017 | 0.652 ± 0.014 | [0.636, 0.669] |
| team_cnn | both | 0.712 ± 0.024 | 0.717 ± 0.022 | [0.690, 0.744] |
| team_rnn | both | 0.594 ± 0.023 | 0.597 ± 0.024 | [0.568, 0.627] |
| cnnlstm | input | 0.262 ± 0.007 | 0.249 ± 0.015 | [0.231, 0.268] |
| fusion | input | 0.254 ± 0.016 | 0.247 ± 0.018 | [0.225, 0.269] |
| ensemble | input | 0.263 ± 0.007 | 0.253 ± 0.003 | [0.249, 0.257] |
| team_cnn | input | 0.256 ± 0.009 | 0.252 ± 0.007 | [0.243, 0.261] |
| team_rnn | input | 0.264 ± 0.017 | 0.257 ± 0.019 | [0.233, 0.281] |

Ensemble members on their own (macro-F1):

| ensemble run | cnn | rnn |
| --- | --- | --- |
| ensemble/both | 0.679 ± 0.031 | 0.604 ± 0.031 |
| ensemble/input | 0.236 ± 0.017 | 0.255 ± 0.011 |
| ensemble/output | 0.681 ± 0.045 | 0.629 ± 0.052 |

## Paired comparisons (A − B, macro-F1)

Same seed means same test rows. *Wins* = seeds where A beat B. *McNemar* = seeds where per-row correctness differed at p < 0.05.

| A | B | mean diff ± std | wins | paired t p | McNemar p<0.05 |
| --- | --- | --- | --- | --- | --- |
| cnnlstm/output | fusion/output | +0.049 ± 0.011 | 5/5 | 0.00058 | 5/5 |
| cnnlstm/output | ensemble/output | +0.040 ± 0.033 | 4/5 | 0.0562 | 3/5 |
| cnnlstm/output | team_cnn/output | +0.007 ± 0.011 | 4/5 | 0.259 | 0/5 |
| cnnlstm/output | team_rnn/output | +0.124 ± 0.033 | 5/5 | 0.0011 | 5/5 |
| fusion/output | ensemble/output | -0.009 ± 0.033 | 3/5 | 0.584 | 2/5 |
| fusion/output | team_cnn/output | -0.042 ± 0.017 | 0/5 | 0.0049 | 4/5 |
| fusion/output | team_rnn/output | +0.075 ± 0.028 | 5/5 | 0.00387 | 5/5 |
| ensemble/output | team_cnn/output | -0.033 ± 0.036 | 2/5 | 0.11 | 3/5 |
| ensemble/output | team_rnn/output | +0.084 ± 0.055 | 5/5 | 0.0273 | 5/5 |
| team_cnn/output | team_rnn/output | +0.117 ± 0.029 | 5/5 | 0.000843 | 5/5 |
| cnnlstm/both | fusion/both | +0.048 ± 0.010 | 5/5 | 0.000388 | 5/5 |
| cnnlstm/both | ensemble/both | +0.047 ± 0.020 | 5/5 | 0.00574 | 4/5 |
| cnnlstm/both | team_cnn/both | -0.017 ± 0.010 | 0/5 | 0.0194 | 1/5 |
| cnnlstm/both | team_rnn/both | +0.102 ± 0.017 | 5/5 | 0.000167 | 5/5 |
| fusion/both | ensemble/both | -0.001 ± 0.016 | 2/5 | 0.883 | 1/5 |
| fusion/both | team_cnn/both | -0.065 ± 0.016 | 0/5 | 0.000778 | 5/5 |
| fusion/both | team_rnn/both | +0.054 ± 0.015 | 5/5 | 0.00146 | 5/5 |
| ensemble/both | team_cnn/both | -0.064 ± 0.017 | 0/5 | 0.000984 | 5/5 |
| ensemble/both | team_rnn/both | +0.055 ± 0.016 | 5/5 | 0.00169 | 5/5 |
| team_cnn/both | team_rnn/both | +0.119 ± 0.018 | 5/5 | 0.000126 | 5/5 |
| cnnlstm/input | fusion/input | +0.002 ± 0.020 | 2/5 | 0.823 | 1/5 |
| cnnlstm/input | ensemble/input | -0.003 ± 0.017 | 2/5 | 0.678 | 0/5 |
| cnnlstm/input | team_cnn/input | -0.003 ± 0.021 | 2/5 | 0.806 | 0/5 |
| cnnlstm/input | team_rnn/input | -0.008 ± 0.028 | 2/5 | 0.565 | 0/5 |
| fusion/input | ensemble/input | -0.006 ± 0.020 | 2/5 | 0.569 | 1/5 |
| fusion/input | team_cnn/input | -0.005 ± 0.018 | 3/5 | 0.598 | 1/5 |
| fusion/input | team_rnn/input | -0.010 ± 0.012 | 1/5 | 0.124 | 1/5 |
| ensemble/input | team_cnn/input | +0.001 ± 0.006 | 3/5 | 0.767 | 0/5 |
| ensemble/input | team_rnn/input | -0.005 ± 0.021 | 3/5 | 0.649 | 0/5 |
| team_cnn/input | team_rnn/input | -0.005 ± 0.016 | 3/5 | 0.484 | 0/5 |
| cnnlstm/output | cnnlstm/both | +0.028 ± 0.013 | 5/5 | 0.00782 | 4/5 |
| cnnlstm/output | cnnlstm/input | +0.479 ± 0.015 | 5/5 | 2.47e-07 | 5/5 |
| cnnlstm/both | cnnlstm/input | +0.450 ± 0.026 | 5/5 | 2.82e-06 | 5/5 |
| fusion/output | fusion/both | +0.028 ± 0.007 | 5/5 | 0.00102 | 3/5 |
| fusion/output | fusion/input | +0.432 ± 0.023 | 5/5 | 1.89e-06 | 5/5 |
| fusion/both | fusion/input | +0.404 ± 0.024 | 5/5 | 2.96e-06 | 5/5 |
| ensemble/output | ensemble/both | +0.036 ± 0.039 | 3/5 | 0.109 | 3/5 |
| ensemble/output | ensemble/input | +0.435 ± 0.040 | 5/5 | 1.63e-05 | 5/5 |
| ensemble/both | ensemble/input | +0.400 ± 0.017 | 5/5 | 7.58e-07 | 5/5 |
| team_cnn/output | team_cnn/both | +0.005 ± 0.009 | 4/5 | 0.321 | 0/5 |
| team_cnn/output | team_cnn/input | +0.470 ± 0.015 | 5/5 | 2.6e-07 | 5/5 |
| team_cnn/both | team_cnn/input | +0.465 ± 0.023 | 5/5 | 1.51e-06 | 5/5 |
| team_rnn/output | team_rnn/both | +0.007 ± 0.043 | 4/5 | 0.75 | 3/5 |
| team_rnn/output | team_rnn/input | +0.347 ± 0.027 | 5/5 | 9.27e-06 | 5/5 |
| team_rnn/both | team_rnn/input | +0.340 ± 0.031 | 5/5 | 1.7e-05 | 5/5 |
