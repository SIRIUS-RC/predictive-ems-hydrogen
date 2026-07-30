# RF and EMS Predictive

Short-term solar power prediction in a residential PV-hydrogen energy management system using Random Forest and SHAP-based feature selection.

## Overview

This project contains a Python pipeline for forecasting grid export power in a residential photovoltaic-hydrogen system. It trains a Random Forest model, applies SHAP-based feature selection, evaluates performance, generates diagnostic plots, and runs a retrospective energy management system (EMS) simulation comparing reactive and predictive control strategies.

## Main Features

- Loads and cleans the dataset.
- Performs a chronological 80/20 train-test split.
- Trains an initial Random Forest regressor.
- Uses SHAP values for feature selection.
- Retrains the final model on the selected features.
- Evaluates model performance using R², MAE, and RMSE.
- Generates plots for:
  - feature importance,
  - actual vs predicted values,
  - residuals,
  - time-series comparison,
  - EMS simulation results.
- Compares reactive and predictive electrolyzer control.

## Dataset

The script expects a dataset named:

```text
HY2RES202404.csv
