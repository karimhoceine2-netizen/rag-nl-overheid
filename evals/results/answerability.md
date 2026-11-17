# Answerability gate

112 reviewed questions (74 answerable, 38 unanswerable); train 84, test 28 (stratified, seed 42). Positive class = unanswerable.

## Cross-validation on the training part (5x5-fold, mean ± sd)

| model | precision | recall | F1 | balanced acc. | answerable blocked |
|---|---|---|---|---|---|
| majority | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.00 | 0.00 ± 0.00 |
| dense_threshold | 0.63 ± 0.18 | 0.67 ± 0.24 | 0.64 ± 0.20 | 0.74 ± 0.14 | 0.19 ± 0.10 |
| logistic | 0.59 ± 0.17 | 0.72 ± 0.17 | 0.64 ± 0.15 | 0.73 ± 0.12 | 0.27 ± 0.15 |

## Test set (used once)

| model | precision | recall | F1 (95% CI) | balanced acc. | answerable blocked | confusion [[TU, FA], [FU, TA]] |
|---|---|---|---|---|---|---|
| majority | 0.00 | 0.00 | 0.00 (0.00-0.00) | 0.50 | 0.00 | [[0, 10], [0, 18]] |
| dense_threshold | 0.67 | 0.80 | 0.73 (0.47-0.91) | 0.79 | 0.22 | [[8, 2], [4, 14]] |
| logistic | 0.73 | 0.80 | 0.76 (0.50-0.94) | 0.82 | 0.17 | [[8, 2], [3, 15]] |

Confusion rows = true class (unanswerable, answerable), columns = predicted. TU = unanswerable caught, FA = unanswerable let through, FU = answerable wrongly blocked, TA = answerable let through.

## Test errors of the logistic model

| id | true label | question | dense_top1 | dense_gap | bm25_top1 | bm25_gap |
|---|---|---|---|---|---|---|
| gu003 | unanswerable | Wat is het telefoonnummer van de Uitvoeringsorganisatie Herstel Toeslagen? | 0.899 | 0.037 | 17.262 | 3.554 |
| ga019 | answerable | Wie krijgt de huurwoning meestal als de reactietermijn van de corporatie voorbij is? | 0.864 | 0.01 | 13.365 | 5.216 |
| ga026 | answerable | Wat is de maximale gevangenisstraf voor kraken zonder geweld? | 0.866 | 0.06 | 21.681 | 20.285 |
| ga001 | answerable | Wanneer wordt de aanvullende tegemoetkoming voor de gemiste basisbeurs uitbetaald? | 0.858 | 0.031 | 21.04 | 10.697 |
| gu011 | unanswerable | Met hoeveel procent mag de huur van een woonbootligplaats in 2027 maximaal stijgen? | 0.897 | 0.021 | 15.996 | 7.753 |

Dense threshold chosen on train: 0.870.
Generated 2026-11-17T09:58:47+00:00 at commit `abdae86`.
Reproduce: `uv run python -m evals.answerability`
