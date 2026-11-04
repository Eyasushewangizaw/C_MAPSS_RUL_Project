"""Three follow-up C-MAPSS experiments.

Operating-condition normalization, a linear health index, and
health-index trajectory matching. The engine split, RUL cap, model
settings, and official test protocol match notebook 02.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

RANDOM_STATE = 42
RUL_CAP = 125
HEALTHY_RUL = 125
LATE_RUL = 15
MIN_HEALTHY_ROWS = 30
STD_FLOOR = 1e-6
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MODEL_NAMES = ["Linear Regression", "SVR", "Random Forest", "XGBoost"]
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
RAW_COLUMNS = ["unit", "cycle", *OP_COLS] + [f"sensor_{i}" for i in range(1, 22)]


def project_paths():
    cwd = Path.cwd()
    if (cwd / "processed_data" / "retained_features.json").exists():
        root = cwd
    elif (cwd.parent / "processed_data" / "retained_features.json").exists():
        root = cwd.parent
    else:
        raise FileNotFoundError("Could not find processed_data/retained_features.json")
    figures = root / "results" / "advanced_figures"
    figures.mkdir(parents=True, exist_ok=True)
    return root, root / "processed_data", root / "data", root / "results", figures


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(y_true, y_pred):
    """NASA/PHM score. d = predicted RUL - true RUL. Lower is better."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    with np.errstate(over="ignore"):
        terms = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    total = float(np.sum(terms))
    if not np.isfinite(total):
        return float("inf")
    return total


def evaluate(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "NASA": nasa_score(y_true, y_pred),
    }


def load_resources():
    root, processed, data, results, figures = project_paths()
    retained = json.loads((processed / "retained_features.json").read_text(encoding="utf-8"))
    selected = json.loads((results / "selected_hyperparameters.json").read_text(encoding="utf-8"))
    baseline = pd.read_csv(results / "model_comparison.csv")
    return {
        "root": root,
        "processed": processed,
        "data": data,
        "results": results,
        "figures": figures,
        "retained": retained,
        "selected": selected,
        "baseline": baseline,
    }


def sensor_columns(retained, subset):
    return [column for column in retained[subset] if column.startswith("sensor_")]


def setting_columns(retained, subset):
    return [column for column in retained[subset] if column.startswith("op_setting")]


def load_split(processed, subset):
    train_df = pd.read_csv(processed / f"train_{subset}_split.csv")
    val_df = pd.read_csv(processed / f"val_{subset}_split.csv")
    overlap = set(train_df["unit"]).intersection(val_df["unit"])
    if overlap:
        raise ValueError(f"{subset} has engines in both splits: {sorted(overlap)[:5]}")
    return train_df, val_df


def load_test_trajectories(data, processed, subset):
    """Full test trajectories. Official RUL is attached only to the last cycle."""
    test_df = pd.read_csv(
        data / f"test_{subset}.txt",
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    )
    if subset == "FD001":
        rul_path = processed / "y_test_FD001_full.csv"
    else:
        rul_path = processed / f"y_test_{subset}.csv"
    official_rul = pd.read_csv(rul_path)["RUL"].to_numpy()
    last = (
        test_df.sort_values(["unit", "cycle"])
        .groupby("unit", sort=True)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )
    expected_units = list(range(1, len(official_rul) + 1))
    if last["unit"].tolist() != expected_units:
        raise ValueError(f"{subset} test units are not 1..N in order")
    test_df = test_df.copy()
    test_df["RUL"] = np.nan
    last_index = (
        test_df.sort_values(["unit", "cycle"])
        .groupby("unit", sort=True)
        .tail(1)
        .sort_values("unit")
        .index
    )
    test_df.loc[last_index, "RUL"] = official_rul
    return test_df


def fit_regime_model(train_df, n_regimes):
    """Cluster operating settings on the training split only."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(train_df[OP_COLS])
    if n_regimes == 1:
        return {"scaler": scaler, "kmeans": None, "n_regimes": 1}
    model = KMeans(n_clusters=n_regimes, random_state=RANDOM_STATE, n_init=10)
    model.fit(scaled)
    return {"scaler": scaler, "kmeans": model, "n_regimes": n_regimes}


def regime_centers(regime_model):
    if regime_model["kmeans"] is None:
        return None
    return regime_model["scaler"].inverse_transform(regime_model["kmeans"].cluster_centers_)


def assign_regimes(df, regime_model):
    out = df.copy()
    scaled = regime_model["scaler"].transform(out[OP_COLS])
    if regime_model["kmeans"] is None:
        out["regime"] = 0
    else:
        out["regime"] = regime_model["kmeans"].predict(scaled)
    return out


def fit_healthy_reference(train_df, sensors):
    """Mean and std of each sensor inside each regime, using healthy cycles.

    Healthy means uncapped RUL is still at or above the 125-cycle knee
    already used by the baseline. A regime with too few such cycles falls
    back to all of its training cycles.
    """
    reference = {}
    for regime, group in train_df.groupby("regime"):
        healthy = group[group["RUL"] >= HEALTHY_RUL]
        used_fallback = False
        if len(healthy) < MIN_HEALTHY_ROWS:
            healthy = group
            used_fallback = True
        stds = healthy[sensors].std(ddof=0)
        reference[int(regime)] = {
            "mean": healthy[sensors].mean(),
            "std": stds,
            "n_healthy": int(len(healthy)),
            "n_regime": int(len(group)),
            "used_fallback": used_fallback,
        }
    return reference


def apply_normalization(df, sensors, reference):
    out = df.copy()
    for sensor in sensors:
        out[f"norm_{sensor}"] = np.nan
    for regime, index in out.groupby("regime").groups.items():
        stats = reference[int(regime)]
        values = out.loc[index, sensors]
        std = stats["std"].astype(float).copy()
        std[std < STD_FLOOR] = np.nan
        adjusted = (values - stats["mean"]) / std
        adjusted = adjusted.fillna(0.0)
        for sensor in sensors:
            out.loc[index, f"norm_{sensor}"] = adjusted[sensor].to_numpy()
    if out[[f"norm_{sensor}" for sensor in sensors]].isna().any().any():
        missing = sorted(set(out["regime"]) - set(reference))
        raise ValueError(f"Regimes missing a training reference: {missing}")
    return out


def model_matrix(df, sensors, settings):
    columns = [f"norm_{sensor}" for sensor in sensors] + list(settings)
    return df[columns]


def make_model(model_name, params):
    if model_name == "Linear Regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ])
    if model_name == "SVR":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", SVR(cache_size=1000, **params)),
        ])
    if model_name == "Random Forest":
        kwargs = dict(params)
        if kwargs.get("max_depth") is None:
            kwargs.pop("max_depth", None)
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **kwargs)
    if model_name == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
    raise ValueError(model_name)


def fit_and_score(model_name, params, X_train, y_train, X_val, y_val, X_test, y_test):
    model = make_model(model_name, params)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    row = {
        "model": model_name,
        "fit_seconds": elapsed,
        "test_pred_min": float(np.min(test_pred)),
        "test_pred_max": float(np.max(test_pred)),
    }
    row.update({f"val_{key}": value for key, value in evaluate(y_val, val_pred).items()})
    row.update({f"test_{key}": value for key, value in evaluate(y_test, test_pred).items()})
    capped_test = np.minimum(y_test, RUL_CAP)
    row.update({
        f"test_capped_{key}": value
        for key, value in evaluate(capped_test, test_pred).items()
    })
    return row


def prepare_subset(bundle, subset):
    train_df, val_df = load_split(bundle["processed"], subset)
    test_df = load_test_trajectories(bundle["data"], bundle["processed"], subset)
    sensors = sensor_columns(bundle["retained"], subset)
    settings = setting_columns(bundle["retained"], subset)
    regime_model = fit_regime_model(train_df, N_REGIMES[subset])
    train_df = assign_regimes(train_df, regime_model)
    val_df = assign_regimes(val_df, regime_model)
    test_df = assign_regimes(test_df, regime_model)
    reference = fit_healthy_reference(train_df, sensors)
    train_df = apply_normalization(train_df, sensors, reference)
    val_df = apply_normalization(val_df, sensors, reference)
    test_df = apply_normalization(test_df, sensors, reference)
    return {
        "subset": subset,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "sensors": sensors,
        "settings": settings,
        "reference": reference,
        "regime_model": regime_model,
        "n_regimes": N_REGIMES[subset],
    }


def supervised_frames(prepared):
    sensors = prepared["sensors"]
    settings = prepared["settings"]
    train_df = prepared["train_df"]
    val_df = prepared["val_df"]
    test_last = (
        prepared["test_df"].sort_values(["unit", "cycle"])
        .groupby("unit", sort=True)
        .tail(1)
        .sort_values("unit")
    )
    if test_last["RUL"].isna().any():
        raise ValueError(f"{prepared['subset']} is missing official test RUL")
    return {
        "X_train": model_matrix(train_df, sensors, settings),
        "y_train": np.minimum(train_df["RUL"].to_numpy(), RUL_CAP),
        "X_val": model_matrix(val_df, sensors, settings),
        "y_val": np.minimum(val_df["RUL"].to_numpy(), RUL_CAP),
        "X_test": model_matrix(test_last, sensors, settings),
        "y_test": test_last["RUL"].to_numpy(),
    }


def run_health_experiment(bundle, prepared, models):
    """Predict capped RUL from the health index alone."""
    enriched, frames = health_frames(prepared)
    rows = []
    subset = prepared["subset"]
    print(
        f"  health index | {subset} | early anchors {frames['n_early']} | "
        f"late anchors {frames['n_late']}",
        flush=True,
    )
    for model_name in models:
        params = bundle["selected"][subset][model_name]
        print(f"  Health index | {subset} | {model_name}", flush=True)
        row = fit_and_score(
            model_name,
            params,
            frames["X_train"],
            frames["y_train"],
            frames["X_val"],
            frames["y_val"],
            frames["X_test"],
            frames["y_test"],
        )
        row.update({"dataset": subset, "experiment": "Health index"})
        print(
            f"    val RMSE {row['val_RMSE']:.2f} | test RMSE {row['test_RMSE']:.2f} | "
            f"{row['fit_seconds']:.1f}s",
            flush=True,
        )
        rows.append(row)
    return enriched, rows


def run_supervised_experiment(bundle, prepared, experiment, models):
    frames = supervised_frames(prepared)
    rows = []
    subset = prepared["subset"]
    for model_name in models:
        params = bundle["selected"][subset][model_name]
        print(f"  {experiment} | {subset} | {model_name}", flush=True)
        row = fit_and_score(
            model_name,
            params,
            frames["X_train"],
            frames["y_train"],
            frames["X_val"],
            frames["y_val"],
            frames["X_test"],
            frames["y_test"],
        )
        row.update({"dataset": subset, "experiment": experiment})
        print(
            f"    val RMSE {row['val_RMSE']:.2f} | test RMSE {row['test_RMSE']:.2f} | "
            f"{row['fit_seconds']:.1f}s",
            flush=True,
        )
        rows.append(row)
    return rows


def fit_health_index(train_df, sensors):
    """Linear health index fit on training anchors only.

    Cycles at or above the 125-cycle knee are the healthy reference (1).
    Cycles with 15 or fewer cycles left are the failed reference (0).
    This follows the piecewise-RUL knee already used in the baseline.
    It is a linear implementation choice, not a reproduction of a
    published health-index formula.
    """
    early = train_df["RUL"] >= HEALTHY_RUL
    late = train_df["RUL"] <= LATE_RUL
    anchors = train_df.loc[early | late].copy()
    if anchors.empty:
        raise ValueError("No health-index anchor cycles were found")
    target = np.where(anchors["RUL"].to_numpy() <= LATE_RUL, 0.0, 1.0)
    features = anchors[[f"norm_{sensor}" for sensor in sensors]]
    model = LinearRegression()
    model.fit(features, target)
    return model, int(early.sum()), int(late.sum())


def add_health_index(df, model, sensors):
    features = df[[f"norm_{sensor}" for sensor in sensors]]
    out = df.copy()
    out["health_index"] = np.clip(model.predict(features), 0.0, 1.0)
    return out


def health_frames(prepared):
    model, n_early, n_late = fit_health_index(prepared["train_df"], prepared["sensors"])
    train_df = add_health_index(prepared["train_df"], model, prepared["sensors"])
    val_df = add_health_index(prepared["val_df"], model, prepared["sensors"])
    test_df = add_health_index(prepared["test_df"], model, prepared["sensors"])
    test_last = (
        test_df.sort_values(["unit", "cycle"])
        .groupby("unit", sort=True)
        .tail(1)
        .sort_values("unit")
    )
    prepared = dict(prepared)
    prepared.update({"train_df": train_df, "val_df": val_df, "test_df": test_df, "hi_model": model})
    frames = {
        "X_train": train_df[["health_index"]],
        "y_train": np.minimum(train_df["RUL"].to_numpy(), RUL_CAP),
        "X_val": val_df[["health_index"]],
        "y_val": np.minimum(val_df["RUL"].to_numpy(), RUL_CAP),
        "X_test": test_last[["health_index"]],
        "y_test": test_last["RUL"].to_numpy(),
        "n_early": n_early,
        "n_late": n_late,
    }
    return prepared, frames


def engine_library(train_df):
    library = []
    for unit, group in train_df.groupby("unit", sort=True):
        ordered = group.sort_values("cycle")
        library.append({
            "unit": int(unit),
            "health_index": ordered["health_index"].to_numpy(dtype=float),
            "rul": ordered["RUL"].to_numpy(dtype=float),
        })
    return library


def best_window_match(query, series, window):
    width = min(window, len(query), len(series["health_index"]))
    if width < 5:
        return None
    query_tail = query[-width:]
    windows = np.lib.stride_tricks.sliding_window_view(series["health_index"], width)
    distances = np.sqrt(np.mean((windows - query_tail) ** 2, axis=1))
    position = int(np.argmin(distances))
    end = position + width - 1
    return (
        float(distances[position]),
        float(series["rul"][end]),
        int(series["unit"]),
        width,
        position,
    )


def match_rul(query, library, window, neighbors):
    found = []
    for series in library:
        matched = best_window_match(query, series, window)
        if matched is not None:
            found.append(matched)
    if not found:
        raise ValueError("No training trajectory was long enough to match")
    found.sort(key=lambda item: item[0])
    chosen = found[:neighbors]
    distances = np.array([item[0] for item in chosen], dtype=float)
    ruls = np.array([item[1] for item in chosen], dtype=float)
    weights = 1.0 / (distances + 1e-6)
    prediction = float(np.sum(weights * ruls) / np.sum(weights))
    return prediction, chosen


def similarity_validation_rows(val_df, library, window, neighbors):
    y_true = []
    y_pred = []
    fractions = (0.50, 0.70, 0.85)
    for _, group in val_df.groupby("unit", sort=True):
        ordered = group.sort_values("cycle")
        health = ordered["health_index"].to_numpy(dtype=float)
        rul = ordered["RUL"].to_numpy(dtype=float)
        length = len(ordered)
        for fraction in fractions:
            end = min(length - 1, max(4, int(round(fraction * length)) - 1))
            prediction, _ = match_rul(health[: end + 1], library, window, neighbors)
            y_true.append(min(float(rul[end]), RUL_CAP))
            y_pred.append(prediction)
    return np.array(y_true), np.array(y_pred)


def run_similarity(prepared, windows=(20, 40), neighbors=(5, 10)):
    library = engine_library(prepared["train_df"])
    subset = prepared["subset"]
    best = None
    search_rows = []
    for window in windows:
        for neighbor_count in neighbors:
            y_true, y_pred = similarity_validation_rows(
                prepared["val_df"], library, window, neighbor_count
            )
            scores = evaluate(y_true, y_pred)
            row = {
                "dataset": subset,
                "window": window,
                "neighbors": neighbor_count,
                **{f"val_{key}": value for key, value in scores.items()},
            }
            search_rows.append(row)
            print(
                f"  similarity | {subset} | window {window} | k {neighbor_count} | "
                f"val RMSE {scores['RMSE']:.2f}",
                flush=True,
            )
            if best is None or scores["RMSE"] < best["scores"]["RMSE"]:
                best = {
                    "window": window,
                    "neighbors": neighbor_count,
                    "scores": scores,
                }
    test_true = []
    test_pred = []
    examples = []
    ordered_test = prepared["test_df"].sort_values(["unit", "cycle"])
    for unit, group in ordered_test.groupby("unit", sort=True):
        health = group["health_index"].to_numpy(dtype=float)
        official = group["RUL"].dropna()
        if len(official) != 1:
            raise ValueError(f"{subset} unit {unit} does not have one official RUL")
        prediction, chosen = match_rul(
            health, library, best["window"], best["neighbors"]
        )
        test_true.append(float(official.iloc[0]))
        test_pred.append(prediction)
        if len(examples) < 1:
            distance, matched_rul, match_unit, width, position = chosen[0]
            matched_series = next(item for item in library if item["unit"] == match_unit)
            examples.append({
                "unit": int(unit),
                "query_tail": health[-width:],
                "match_hi": matched_series["health_index"][position: position + width],
                "match_unit": match_unit,
                "distance": distance,
                "matched_rul": matched_rul,
                "prediction": prediction,
                "actual": float(official.iloc[0]),
            })
    test_true = np.array(test_true)
    test_pred = np.array(test_pred)
    test_scores = evaluate(test_true, test_pred)
    capped_scores = evaluate(np.minimum(test_true, RUL_CAP), test_pred)
    result = {
        "dataset": subset,
        "experiment": "History matching",
        "model": "Similarity",
        "window": best["window"],
        "neighbors": best["neighbors"],
        "val_RMSE": best["scores"]["RMSE"],
        "val_MAE": best["scores"]["MAE"],
        "val_NASA": best["scores"]["NASA"],
        "test_RMSE": test_scores["RMSE"],
        "test_MAE": test_scores["MAE"],
        "test_NASA": test_scores["NASA"],
        "test_capped_RMSE": capped_scores["RMSE"],
        "test_capped_MAE": capped_scores["MAE"],
        "test_capped_NASA": capped_scores["NASA"],
        "test_pred_min": float(np.min(test_pred)),
        "test_pred_max": float(np.max(test_pred)),
        "fit_seconds": np.nan,
    }
    print(
        f"  selected similarity | {subset} | window {best['window']} | "
        f"k {best['neighbors']} | test RMSE {test_scores['RMSE']:.2f}",
        flush=True,
    )
    return result, search_rows, examples, library


def run_all(models=None, subsets=None):
    """Run all three experiments and checkpoint results after each dataset."""
    bundle = load_resources()
    models = list(models or MODEL_NAMES)
    subsets = list(subsets or SUBSETS)
    log_path = bundle["results"] / "advanced_training_log.txt"
    log_path.write_text("", encoding="utf-8")
    import builtins
    original_print = builtins.print

    def log(message):
        original_print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")

    def tee_print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        original_print(*args, **kwargs)
        text = " ".join(str(arg) for arg in args)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text + "\n")

    builtins.print = tee_print
    try:
        norm_rows = []
        health_rows = []
        sim_rows = []
        sim_search = []
        prepared_by_subset = {}
        examples = {}
        for subset in subsets:
            log("\n" + "=" * 60)
            log(subset)
            log("=" * 60)
            prepared = prepare_subset(bundle, subset)
            norm_rows.extend(run_supervised_experiment(
                bundle, prepared, "Normalization", models
            ))
            enriched, health_part = run_health_experiment(bundle, prepared, models)
            health_rows.extend(health_part)
            similarity, search_part, example, _library = run_similarity(enriched)
            sim_rows.append(similarity)
            sim_search.extend(search_part)
            prepared_by_subset[subset] = enriched
            examples[subset] = example
            _write_checkpoints(bundle["results"], norm_rows, health_rows, sim_rows, sim_search)
            log(f"{subset} checkpoint saved")
    finally:
        builtins.print = original_print
    return {
        "bundle": bundle,
        "norm_rows": norm_rows,
        "health_rows": health_rows,
        "sim_rows": sim_rows,
        "sim_search": sim_search,
        "prepared": prepared_by_subset,
        "examples": examples,
    }


def _write_checkpoints(results_dir, norm_rows, health_rows, sim_rows, sim_search):
    pd.DataFrame(norm_rows).to_csv(results_dir / "advanced_normalization.csv", index=False)
    pd.DataFrame(health_rows).to_csv(results_dir / "advanced_health_index.csv", index=False)
    pd.DataFrame(sim_rows).to_csv(results_dir / "advanced_similarity.csv", index=False)
    pd.DataFrame(sim_search).to_csv(results_dir / "advanced_similarity_search.csv", index=False)
