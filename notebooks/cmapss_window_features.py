"""Window-feature RUL models for NASA C-MAPSS.

Classical pipeline from Galchar, Goel, and Kanu (arXiv:2604.27234):
30-cycle windows, five statistics per sensor, Ridge, polynomial Ridge,
and XGBoost. FD002 and FD004 add regime normalization and causal smoothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

RANDOM_STATE = 42
WINDOW = 30
STD_DROP = 0.01
REGIME_COUNTS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = ["unit", "cycle", *OP_COLS, *SENSOR_COLS]
PAPER_SENSORS = {
    "FD001": [f"sensor_{i}" for i in range(1, 22) if i not in {1, 5, 6, 10, 16, 18, 19}],
    "FD003": [f"sensor_{i}" for i in range(1, 22) if i not in {1, 5, 10, 16, 18, 19}],
}
PAPER_RMSE = {
    ("FD003", "Ridge"): 17.494,
    ("FD003", "Polynomial Ridge"): 16.604,
    ("FD003", "XGBoost"): 13.355,
}


def project_paths():
    cwd = Path.cwd()
    if (cwd / "processed_data" / "retained_features.json").exists():
        root = cwd
    elif (cwd.parent / "processed_data" / "retained_features.json").exists():
        root = cwd.parent
    else:
        raise FileNotFoundError("Could not find processed_data/retained_features.json")
    feature_dir = root / "processed_data" / "window_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    return root, root / "processed_data", root / "data", root / "results", feature_dir


def nasa_score(y_true, y_pred):
    """h = predicted - true. Early uses 13, late uses 10. Lower is better."""
    h = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    with np.errstate(over="ignore"):
        terms = np.where(h < 0, np.exp(-h / 13.0) - 1.0, np.exp(h / 10.0) - 1.0)
    total = float(np.sum(terms))
    return total if np.isfinite(total) else float("inf")


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "NASA": nasa_score(y_true, y_pred),
    }


def load_split_frames(processed, data, subset):
    train_df = pd.read_csv(processed / f"train_{subset}_split.csv")
    val_df = pd.read_csv(processed / f"val_{subset}_split.csv")
    overlap = set(train_df["unit"]).intersection(val_df["unit"])
    if overlap:
        raise ValueError(f"{subset} engine overlap: {sorted(overlap)[:5]}")
    test_df = pd.read_csv(data / f"test_{subset}.txt", sep=r"\s+", header=None, names=RAW_COLUMNS)
    rul_name = "y_test_FD001_full.csv" if subset == "FD001" else f"y_test_{subset}.csv"
    official = pd.read_csv(processed / rul_name)["RUL"].to_numpy(dtype=float)
    last_units = (
        test_df.sort_values(["unit", "cycle"]).groupby("unit", sort=True).tail(1).sort_values("unit")
    )
    if last_units["unit"].tolist() != list(range(1, len(official) + 1)):
        raise ValueError(f"{subset} test units are not 1..N")
    test_df = test_df.copy()
    test_df["official_rul"] = np.nan
    test_df.loc[last_units.index, "official_rul"] = official
    return train_df, val_df, test_df


def select_sensors(train_df):
    """Drop near-constant sensors using the training split only."""
    kept = []
    rows = []
    for sensor in SENSOR_COLS:
        values = train_df[sensor]
        std = float(values.std(ddof=0))
        nunique = int(values.nunique())
        corr = float(values.corr(train_df["RUL"])) if std > 0 else float("nan")
        drop = std < STD_DROP or nunique <= 1
        rows.append({
            "sensor": sensor,
            "std": std,
            "nunique": nunique,
            "corr_rul": corr,
            "dropped": drop,
        })
        if not drop:
            kept.append(sensor)
    return kept, pd.DataFrame(rows)


def piecewise_rul(rul, cap):
    return np.minimum(np.asarray(rul, dtype=float), cap)


def _slope_axis():
    t = np.arange(WINDOW, dtype=float)
    t_centered = t - t.mean()
    denom = float(np.dot(t_centered, t_centered))
    return t_centered, denom


def _stats_from_view(view):
    """view shape (n_windows, n_sensors, window) -> (n_windows, n_sensors * 5)."""
    mean = view.mean(axis=2)
    std = view.std(axis=2)
    last = view[:, :, -1]
    delta = last - view[:, :, 0]
    t_centered, denom = _slope_axis()
    slope = np.tensordot(view, t_centered, axes=([2], [0])) / denom
    block = np.stack([mean, std, last, delta, slope], axis=2)
    return block.reshape(view.shape[0], -1)


def _one_window_stats(window):
    """window shape (length, n_sensors)."""
    view = np.moveaxis(window, 0, 1)[None, :, :]
    return _stats_from_view(view)[0]


def windows_from_frame(df, sensors, window, cap, pad_short=False):
    """One row per window. Training uses only full windows. Test may pad."""
    if window != WINDOW:
        raise ValueError("Window length is fixed at 30 so the slope axis stays aligned")
    features = []
    labels = []
    units = []
    end_cycles = []
    official = []
    regimes = []
    has_regime = "regime" in df.columns
    has_official = "official_rul" in df.columns
    for unit, group in df.groupby("unit", sort=True):
        ordered = group.sort_values("cycle")
        values = ordered[sensors].to_numpy(dtype=float)
        length = len(ordered)
        if pad_short:
            if length >= window:
                segment = values[-window:]
            else:
                pad = np.zeros((window - length, len(sensors)), dtype=float)
                segment = np.vstack([pad, values])
            features.append(_one_window_stats(segment))
            units.append(int(unit))
            end_cycles.append(int(ordered["cycle"].iloc[-1]))
            if has_official:
                official.append(float(ordered["official_rul"].dropna().iloc[0]))
            if has_regime:
                regimes.append(int(ordered["regime"].iloc[-1]))
            continue
        if length < window or "RUL" not in ordered.columns:
            continue
        view = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
        stats = _stats_from_view(view)
        features.append(stats)
        rul = ordered["RUL"].to_numpy(dtype=float)
        labels.append(piecewise_rul(rul[window - 1 :], cap))
        units.append(np.full(stats.shape[0], int(unit)))
        end_cycles.append(ordered["cycle"].to_numpy()[window - 1 :])
        if has_regime:
            regimes.append(ordered["regime"].to_numpy()[window - 1 :])
    if pad_short:
        out = {
            "X": np.vstack(features),
            "unit": np.asarray(units),
            "end_cycle": np.asarray(end_cycles),
        }
        if official:
            out["official_rul"] = np.asarray(official, dtype=float)
        if regimes:
            out["regime"] = np.asarray(regimes)
        return out
    if not features:
        raise ValueError("No full windows were created")
    out = {
        "X": np.vstack(features),
        "y": np.concatenate(labels),
        "unit": np.concatenate(units),
        "end_cycle": np.concatenate(end_cycles),
    }
    if regimes:
        out["regime"] = np.concatenate(regimes)
    return out


def feature_names(sensors):
    names = []
    for sensor in sensors:
        names.extend([
            f"{sensor}_mean",
            f"{sensor}_std",
            f"{sensor}_last",
            f"{sensor}_delta",
            f"{sensor}_slope",
        ])
    return names


def fit_regime_model(train_df, n_regimes):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_df[OP_COLS])
    if n_regimes == 1:
        return {"scaler": scaler, "kmeans": None}
    model = KMeans(n_clusters=n_regimes, random_state=RANDOM_STATE, n_init=10)
    model.fit(scaled)
    return {"scaler": scaler, "kmeans": model}


def assign_regime(df, regime_model):
    out = df.copy()
    scaled = regime_model["scaler"].transform(out[OP_COLS])
    if regime_model["kmeans"] is None:
        out["regime"] = 0
    else:
        out["regime"] = regime_model["kmeans"].predict(scaled)
    return out


def fit_regime_sensor_stats(train_df, sensors):
    stats = {}
    for regime, group in train_df.groupby("regime"):
        std = group[sensors].std(ddof=0).astype(float)
        std[std < 1e-6] = np.nan
        stats[int(regime)] = {"mean": group[sensors].mean(), "std": std}
    return stats


def apply_regime_norm(df, sensors, stats):
    out = df.copy()
    normalized = pd.DataFrame(index=out.index, columns=sensors, dtype=float)
    for regime, index in out.groupby("regime").groups.items():
        center = stats[int(regime)]
        values = (out.loc[index, sensors] - center["mean"]) / center["std"]
        normalized.loc[index, sensors] = values.fillna(0.0).to_numpy()
    out[sensors] = normalized.astype(float)
    return out


def causal_smooth(df, sensors, alpha):
    """Exponential smooth each engine forward in time. No future values."""
    out = df.copy()
    pieces = []
    for _, group in out.groupby("unit", sort=False):
        ordered = group.sort_values("cycle")
        values = ordered[sensors].to_numpy(dtype=float)
        smoothed = np.empty_like(values)
        smoothed[0] = values[0]
        for t in range(1, len(values)):
            smoothed[t] = alpha * values[t] + (1.0 - alpha) * smoothed[t - 1]
        updated = ordered.copy()
        updated[sensors] = smoothed
        pieces.append(updated)
    return pd.concat(pieces).sort_index()


def make_models():
    ridge = Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    poly = Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",
        early_stopping_rounds=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return {
        "Ridge": ridge,
        "Polynomial Ridge": poly,
        "XGBoost": xgb,
    }


def fit_model(name, model, X_train, y_train, X_val, y_val):
    if name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


def append_regime_feature(pack):
    if "regime" not in pack:
        return pack
    pack = dict(pack)
    pack["X"] = np.column_stack([pack["X"], pack["regime"].astype(float)])
    return pack


def prepare_frames(train_df, val_df, test_df, sensors, n_regimes, use_regime, alpha):
    """Fit regime statistics on training cycles only, then smooth causally."""
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()
    if use_regime:
        regime_model = fit_regime_model(train, n_regimes)
        train = assign_regime(train, regime_model)
        val = assign_regime(val, regime_model)
        test = assign_regime(test, regime_model)
        sensor_stats = fit_regime_sensor_stats(train, sensors)
        train = apply_regime_norm(train, sensors, sensor_stats)
        val = apply_regime_norm(val, sensors, sensor_stats)
        test = apply_regime_norm(test, sensors, sensor_stats)
    if alpha is not None:
        train = causal_smooth(train, sensors, alpha)
        val = causal_smooth(val, sensors, alpha)
        test = causal_smooth(test, sensors, alpha)
    return train, val, test


def run_condition(train_df, val_df, test_df, sensors, cap, use_regime_feature, model_names):
    train_w = windows_from_frame(train_df, sensors, WINDOW, cap, pad_short=False)
    val_w = windows_from_frame(val_df, sensors, WINDOW, cap, pad_short=False)
    test_w = windows_from_frame(test_df, sensors, WINDOW, cap, pad_short=True)
    if use_regime_feature:
        train_w = append_regime_feature(train_w)
        val_w = append_regime_feature(val_w)
        test_w = append_regime_feature(test_w)
    rows = []
    predictions = {}
    for name in model_names:
        model = make_models()[name]
        fit_model(name, model, train_w["X"], train_w["y"], val_w["X"], val_w["y"])
        val_pred = model.predict(val_w["X"])
        test_pred = model.predict(test_w["X"])
        val_scores = evaluate(val_w["y"], val_pred)
        official_scores = evaluate(test_w["official_rul"], test_pred)
        capped_scores = evaluate(np.minimum(test_w["official_rul"], cap), test_pred)
        rows.append({
            "model": name,
            "n_train_windows": int(len(train_w["y"])),
            "n_val_windows": int(len(val_w["y"])),
            "n_features": int(train_w["X"].shape[1]),
            "val_RMSE": val_scores["RMSE"],
            "val_MAE": val_scores["MAE"],
            "val_R2": val_scores["R2"],
            "val_NASA": val_scores["NASA"],
            **{f"test_{key}": value for key, value in official_scores.items()},
            **{f"test_capped_{key}": value for key, value in capped_scores.items()},
        })
        predictions[name] = {
            "unit": test_w["unit"],
            "y_true": test_w["official_rul"],
            "y_pred": np.asarray(test_pred, dtype=float),
        }
    return rows, predictions, (train_w, val_w, test_w)


ALPHAS = (0.2, 0.4, 0.6)
MODEL_NAMES = ("Ridge", "Polynomial Ridge", "XGBoost")


def _print_row(row):
    print(
        f"  {row['condition']:16} {row['model']:18} "
        f"val {row['val_RMSE']:.3f} | official {row['test_RMSE']:.3f} "
        f"MAE {row['test_MAE']:.3f} R2 {row['test_R2']:.3f} NASA {row['test_NASA']:.1f} "
        f"| capped-target {row['test_capped_RMSE']:.3f}",
        flush=True,
    )


def run_experiments():
    """Fit every model on training engines. Alpha uses validation RMSE only."""
    _, processed, data, results, feature_dir = project_paths()
    all_rows = []
    alpha_rows = []
    chosen_predictions = {}
    for subset in SUBSETS:
        print(f"\n===== {subset} =====", flush=True)
        train_raw, val_raw, test_raw = load_split_frames(processed, data, subset)
        sensors, audit = select_sensors(train_raw)
        audit.to_csv(results / f"window_sensor_audit_{subset}.csv", index=False)
        dropped = audit.loc[audit["dropped"], "sensor"].tolist()
        print(f"dropped {dropped}", flush=True)
        print(f"kept {len(sensors)} sensors -> {len(sensors) * 5} window features", flush=True)
        n_regimes = REGIME_COUNTS[subset]
        for cap in (125, 130):
            print(f"-- cap {cap}", flush=True)
            alpha_choice = {}
            for use_regime, label in ((False, "smooth"), (True, "regime_smooth")):
                best_alpha = None
                best_val = float("inf")
                for alpha in ALPHAS:
                    frames = prepare_frames(
                        train_raw, val_raw, test_raw, sensors, n_regimes, use_regime, alpha
                    )
                    rows, _, _ = run_condition(
                        *frames, sensors, cap, use_regime, model_names=["Ridge"]
                    )
                    val_rmse = rows[0]["val_RMSE"]
                    alpha_rows.append({
                        "subset": subset,
                        "cap": cap,
                        "base": label,
                        "alpha": alpha,
                        "val_RMSE": val_rmse,
                    })
                    print(f"  alpha search {label} a={alpha:.1f} ridge val {val_rmse:.3f}", flush=True)
                    if val_rmse < best_val:
                        best_val = val_rmse
                        best_alpha = alpha
                alpha_choice[label] = best_alpha
            conditions = [
                ("window_only", False, None),
                ("regime", True, None),
                ("smooth", False, alpha_choice["smooth"]),
                ("regime_smooth", True, alpha_choice["regime_smooth"]),
            ]
            for condition, use_regime, alpha in conditions:
                frames = prepare_frames(
                    train_raw, val_raw, test_raw, sensors, n_regimes, use_regime, alpha
                )
                rows, predictions, packs = run_condition(
                    *frames, sensors, cap, use_regime, model_names=list(MODEL_NAMES)
                )
                for row in rows:
                    row.update({
                        "subset": subset,
                        "cap": cap,
                        "condition": condition,
                        "alpha": alpha if alpha is not None else "",
                        "n_sensors": len(sensors),
                        "sensors": ",".join(sensors),
                        "dropped_sensors": ",".join(dropped),
                    })
                    all_rows.append(row)
                    _print_row(row)
                if cap == 125:
                    names = feature_names(sensors)
                    if use_regime:
                        names = names + ["regime_id"]
                    payload = {"feature_names": np.array(names)}
                    for split_name, pack in zip(("train", "val", "test"), packs):
                        payload[f"X_{split_name}"] = pack["X"]
                        payload[f"unit_{split_name}"] = pack["unit"]
                        payload[f"end_cycle_{split_name}"] = pack["end_cycle"]
                    payload["y_train"] = packs[0]["y"]
                    payload["y_val"] = packs[1]["y"]
                    payload["y_test"] = packs[2]["official_rul"]
                    np.savez_compressed(
                        feature_dir / f"{subset}_cap{cap}_{condition}.npz",
                        **payload,
                    )
                chosen_predictions[(subset, cap, condition)] = predictions
            pd.DataFrame(all_rows).to_csv(results / "window_feature_results.csv", index=False)
            pd.DataFrame(alpha_rows).to_csv(results / "window_alpha_search.csv", index=False)
    pred_rows = []
    for (subset, cap, condition), predictions in chosen_predictions.items():
        for model, pack in predictions.items():
            for unit, y_true, y_pred in zip(pack["unit"], pack["y_true"], pack["y_pred"]):
                pred_rows.append({
                    "subset": subset,
                    "cap": cap,
                    "condition": condition,
                    "model": model,
                    "unit": int(unit),
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                })
    pd.DataFrame(pred_rows).to_csv(results / "window_feature_predictions.csv", index=False)
    print("saved", results / "window_feature_results.csv", flush=True)
    return pd.DataFrame(all_rows)
