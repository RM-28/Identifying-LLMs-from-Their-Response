| model | setting | test accuracy | test macro-F1 | acc drop | macro-F1 drop | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| CNN | baseline | 0.682 ± 0.005 | 0.683 ± 0.006 | +0.000 | +0.000 | 3 |
| CNN | no_markdown | 0.661 ± 0.008 | 0.663 ± 0.007 | +0.021 | +0.020 | 3 |
| CNN | truncated_50 | 0.603 ± 0.010 | 0.600 ± 0.016 | +0.079 | +0.083 | 3 |
| CNN | no_whitespace | 0.685 ± 0.002 | 0.684 ± 0.007 | -0.003 | -0.001 | 3 |
| Parallel | baseline | 0.643 ± 0.011 | 0.646 ± 0.005 | +0.000 | +0.000 | 3 |
| Parallel | no_markdown | 0.629 ± 0.009 | 0.630 ± 0.006 | +0.014 | +0.016 | 3 |
| Parallel | truncated_50 | 0.584 ± 0.005 | 0.581 ± 0.008 | +0.059 | +0.065 | 3 |
| Parallel | no_whitespace | 0.651 ± 0.007 | 0.652 ± 0.013 | -0.008 | -0.005 | 3 |
