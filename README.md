# Time Series Deep Learning: A Rigorous Architecture Comparison

A comparative study of five deep learning architectures — **Feed-Forward, RNN, LSTM, CNN, and ResNet** — for single-step stock return forecasting, built around a walk-forward evaluation protocol rather than a single train/test split.

The goal isn't to crown a "best" architecture. It's to test whether any of these models earn their added complexity over a naive baseline on this task — and to report that honestly either way.

## Why this repo is structured the way it is

Most public time-series-DL notebooks compare architectures by training each once on a single split and reporting RMSE on raw price levels. That setup is misleading in a specific way: raw stock prices are dominated by trend, so "predict tomorrow ≈ today" already scores a deceptively low RMSE — any model can look skillful without actually forecasting anything. This project is built to avoid that trap:

| Risk | How it's handled |
|---|---|
| Non-stationary target inflates apparent accuracy | Forecast **log returns**, not price levels — confirmed stationary via Augmented Dickey-Fuller test |
| No way to judge whether a model has real skill | Every model is scored against **two naive baselines** (zero-return and persistence) |
| Single split hides variance across time | **Walk-forward (rolling-origin) validation**, 3 folds |
| Single training run hides seed sensitivity | Each model trained with **3 random seeds** per fold, reported as mean ± std |
| Unfair comparison across architectures | All 5 models consume the **same lookback window**, same scaling, same train/val/test boundaries |
| Data leakage via preprocessing | Scalers fit on **training data only**, per fold — never on the full series up front |
| One metric hides the real story | Report **RMSE, MAE, and directional accuracy** (did the model get the sign of the move right?) |

## What's in the notebook

`Time_Series_Analysis_with_DL_v2.ipynb` runs end-to-end and includes:

1. **Methodology & Threats to Validity** — stated up front, before any results, so the evaluation design can be judged on its own merits.
2. **Data acquisition** — pulls daily adjusted-close prices from the [Alpha Vantage API](https://www.alphavantage.co/documentation/). Falls back to a synthetic geometric Brownian motion series if no API key is set, so the pipeline itself can be verified offline.
3. **Stationarity check** — ADF test comparing log price vs. log returns, justifying the return-based target.
4. **Shared windowing + walk-forward split logic**, used identically by every model.
5. **Five architectures** (Feed-Forward, RNN, LSTM, CNN, ResNet), each a thin wrapper producing a single scalar return prediction from a shared input window.
6. **Training with early stopping** on a held-out validation slice, instead of a fixed epoch count forced across architecturally different models.
7. **Evaluation**: walk-forward × multiple seeds × two naive baselines, summarized as mean ± std per model, with RMSE and directional-accuracy plots.
8. **Discussion & Limitations**, written as prompts to fill in after a real run rather than a canned conclusion.

## Getting started

```bash
pip install -r requirements.txt
export ALPHAVANTAGE_API_KEY="your_key_here"   # free tier: https://www.alphavantage.co/support/#api-key
jupyter notebook Time_Series_Analysis_with_DL_v2.ipynb
```

Without an API key set, the notebook still runs end-to-end against a synthetic fallback series — useful for verifying the pipeline, but **results are only meaningful when run against real market data**.

## Reading the results

The bar for a model to be interesting is **beating both naive baselines outside one standard deviation**, not just posting a lower point-estimate RMSE. If none of the deep models clear that bar, that's a legitimate finding — daily-frequency single-stock returns are close to a random walk, and a study that surfaces that honestly is more useful than one that manufactures a winner.

## Repo structure

```
.
├── Time_Series_Analysis_with_DL_v2.ipynb   # main notebook
├── README.md
└── requirements.txt
```

## Known limitations

- Single ticker — findings shouldn't be generalized to "which architecture wins" without testing on multiple, structurally different series.
- Univariate only — no volume, macro, or sentiment features.
- Point forecasts only — no uncertainty quantification (a natural next step would be quantile or conformal prediction intervals).
- Architecture hyperparameters (hidden size, learning rate, lookback window) were fixed rather than tuned per model, to keep the cross-architecture comparison fair and the study tractable.

## License

MIT
