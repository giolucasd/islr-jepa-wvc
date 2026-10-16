# Evaluation Protocol

## Dataset

LSFB-isol contains ~120,000 clips of isolated signs from Belgian French Sign Language (LSFB), performed by 99 signers across 4,657 sign classes.
The dataset is partitioned into five folds that are fully signer-disjoint: no signer appears in more than one fold. This property is the foundation of the evaluation design.

| Split | Folds | Signers | Clips |
|---|---|---|---|
| train | 2, 3, 4 | 59 | ~68,262 |
| test  | 0, 1    | 40 | ~52,477 |

## Two Evaluation Protocols

### Protocol 1 -- Signer-Independent (SI)

The model is trained on signers from folds 2, 3 and 4 and evaluated on signers from folds 0 and 1. No signer appears in both sets.

This protocol approximates real-world deployment conditions, where a system must generalise to people it has never seen. Performance under this protocol reflects how well a model captures sign-specific features rather than signer-specific ones.

### Protocol 2 -- Signer-Dependent (SD)

Clips from all 99 signers are randomly partitioned into train (40%), validation (20%) and test (40%) sets, sampling per-signer and stratified by class. The same signers appear in all three sets.

This protocol is not realistic for deployment, but it measures the upper-bound performance achievable when signer identity is not a source of distribution shift.


## How the Two Protocols Are Used Together

The primary comparison in this work is not the absolute accuracy of any single model, but the **gap** between its SI and SD performance.

A model that performs well under SD but drops sharply under SI is likely relying on signer-specific cues (appearance, style, body proportions) rather than on the linguistic content of the signs. A model whose performance is more stable across the two protocols has learned representations that generalise across signers.

This framing allows us to use LSFB-isol to address two questions:

1. **Does signer-dependent evaluation overestimate real-world performance?**
   If all models show a consistent SD->SI drop, this confirms that signer-dependent benchmarks are an unreliable proxy for deployment performance -- a finding that motivates the use of signer-independent evaluation as the standard for this dataset.

2. **Which model families are less prone to signer overfitting?**
   Comparing the SI/SD gap across architectures and pretraining strategies (supervised, self-supervised, no pretraining) reveals which approaches produce more signer-invariant representations.

## Reported Metrics

All results report accuracies: **micro top-1**, **macro top-1** and **macro top-5** on the respective test sets, as well as expected calibration error (**ECE**). Macro averaging is used throughout because LSFB-isol is class-imbalanced; micro accuracy alone would over-represent frequent signs and obscure per-class behaviour.
