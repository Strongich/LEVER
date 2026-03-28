"""
Train a regression model to predict reward from policy embeddings.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_training_data(json_path: str = "data/regressor_training_data.json"):
    """
    Load training data from JSON file.

    Args:
        json_path: Path to the JSON file containing training data

    Returns:
        Tuple of (X, y) where X is policy embeddings and y is rewards

    Raises:
        FileNotFoundError: If the training data file doesn't exist
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Training data file not found: {json_path}\n"
            "This means the successor models haven't been trained yet.\n"
            "Please run 'python pi2vec/train_successor.py' first to generate the training data."
        )

    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert lists back to numpy arrays
    X = np.array(data["policy_embedding"])
    y = np.array(data["reward"])

    # Ensure X is 2D: (n_samples, n_features)
    # If it's 3D, flatten the extra dimensions
    if X.ndim > 2:
        # Reshape to 2D: flatten all dimensions except the first
        original_shape = X.shape
        X = X.reshape(original_shape[0], -1)
        print(f"Reshaped policy embeddings from {original_shape} to {X.shape}")

    return X, y


def load_training_data_with_targets(
    json_path: str = "data/regressor_training_data.json",
    embedding_key: str = "policy_embedding",
):
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Training data file not found: {json_path}\n"
            "This means the successor models haven't been trained yet.\n"
            "Please run 'python pi2vec/train_successor.py' first to generate the training data."
        )

    with open(json_path, "r") as f:
        data = json.load(f)

    if embedding_key in data:
        X = np.array(data[embedding_key])
    elif "policy_embedding" in data:
        X = np.array(data["policy_embedding"])
    else:
        raise ValueError(f"Training data missing '{embedding_key}' embeddings.")
    y = np.array(data["reward"])
    targets = data.get("policy_target")
    if targets is None:
        raise ValueError("Training data missing 'policy_target' labels.")
    targets = list(targets)

    if X.ndim > 2:
        original_shape = X.shape
        X = X.reshape(original_shape[0], -1)
        print(f"Reshaped policy embeddings from {original_shape} to {X.shape}")

    return X, y, targets


def load_training_data_multi(
    json_path: str,
    embedding_keys: list[str],
):
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Training data file not found: {json_path}\n"
            "This means the successor models haven't been trained yet.\n"
            "Please run 'python pi2vec/train_successor.py' first to generate the training data."
        )

    with open(json_path, "r") as f:
        data = json.load(f)

    embeddings = {}
    for key in embedding_keys:
        if key in data:
            X = np.array(data[key])
        elif "policy_embedding" in data:
            X = np.array(data["policy_embedding"])
        else:
            raise ValueError(f"Training data missing '{key}' embeddings.")
        if X.ndim > 2:
            original_shape = X.shape
            X = X.reshape(original_shape[0], -1)
            print(f"Reshaped policy embeddings from {original_shape} to {X.shape}")
        embeddings[key] = X

    y = np.array(data["reward"])
    targets = data.get("policy_target")
    if targets is None:
        raise ValueError("Training data missing 'policy_target' labels.")
    targets = list(targets)

    return embeddings, y, targets


def save_training_data(
    json_path: str, embeddings, rewards, policy_targets: list[str] | None = None
):
    policy_embeddings = []
    for embedding in embeddings:
        if isinstance(embedding, list):
            policy_embeddings.append(embedding)
        else:
            policy_embeddings.append(embedding.tolist())
    data = {
        "policy_embedding": policy_embeddings,
        "reward": list(rewards),
    }
    if policy_targets is not None:
        data["policy_target"] = list(policy_targets)
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def _nmae(y_true, y_pred) -> float:
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.mean(np.abs(y_true))
    return mae / denom if denom > 0 else np.inf


def kfold_nmae(
    X,
    y,
    k: int = 10,
    normalize_embeddings: bool = False,
    seed: int = 42,
):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(
            f"  Fold {fold_idx}/{k}: train={len(train_idx)} val={len(val_idx)}"
        )
        model = train_regressor(
            X[train_idx],
            y[train_idx],
            normalize_embeddings=normalize_embeddings,
        )
        y_pred = model.predict(X[val_idx])
        score = _nmae(y[val_idx], y_pred)
        print(f"  Fold {fold_idx}/{k}: NMAE = {score:.4f}")
        scores.append(score)
    avg = float(np.mean(scores)) if scores else float("nan")
    print(f"  Avg NMAE (k={k}): {avg:.4f}")
    return scores, avg


def train_regressor_variants(
    source_json_path: str,
    output_json_paths: dict[str, str],
    output_model_paths: dict[str, str],
    output_plot_paths: dict[str, str] | None = None,
    variants: dict[str, list[str]] | None = None,
    embedding_key_by_variant: dict[str, str] | None = None,
    normalize_embeddings: bool = False,
    random_search: bool = False,
    random_search_iters: int = 25,
    random_search_cv: int = 10,
    random_search_seed: int = 42,
):
    if variants is None:
        variants = {
            "base": ["gold", "path", "hazard", "lever"],
            "pair": ["gold", "path", "hazard", "lever", "path-gold", "hazard-lever"],
            "trip": ["gold", "path", "hazard", "lever", "path-gold-hazard"],
        }
    if embedding_key_by_variant is None:
        embedding_key_by_variant = {
            "base": "policy_embedding_base",
            "pair": "policy_embedding_pair",
            "trip": "policy_embedding_trip",
        }

    embedding_keys = list(set(embedding_key_by_variant.values()))
    embeddings_by_key, y, targets = load_training_data_multi(
        source_json_path, embedding_keys
    )
    if len(y) == 0:
        raise ValueError("No training data found for regressor variants.")

    for key, allowed_targets in variants.items():
        embedding_key = embedding_key_by_variant.get(key, "policy_embedding")
        X = embeddings_by_key[embedding_key]
        mask = [t in allowed_targets for t in targets]
        X_sub = X[mask]
        y_sub = y[mask]
        targets_sub = [t for t in targets if t in allowed_targets]
        if len(X_sub) == 0:
            raise ValueError(f"No samples found for regressor variant '{key}'.")

        output_json = output_json_paths.get(key)
        output_model = output_model_paths.get(key)
        output_plot = output_plot_paths.get(key) if output_plot_paths else None
        if output_json:
            save_training_data(output_json, X_sub, y_sub, targets_sub)
        print(f"[{key}] 10-fold CV NMAE (90/10 splits):")
        kfold_nmae(
            X_sub,
            y_sub,
            k=10,
            normalize_embeddings=normalize_embeddings,
        )
        model = train_regressor(
            X_sub,
            y_sub,
            normalize_embeddings=normalize_embeddings,
            random_search=random_search,
            random_search_iters=random_search_iters,
            random_search_cv=random_search_cv,
            random_search_seed=random_search_seed,
        )
        if output_model:
            save_model(model, model_path=output_model)
        if output_plot:
            y_pred = model.predict(X_sub)
            plot_regression_results(y_sub, y_pred, save_path=output_plot)


def _make_hgbr_pipeline(normalize_embeddings: bool):
    regressor = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=10,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
    if normalize_embeddings:
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", regressor),
                ]
            ),
            "regressor__",
        )
    return regressor, ""


def _hgbr_param_distributions(prefix: str = ""):
    return {
        f"{prefix}max_iter": [200, 300, 500, 800],
        f"{prefix}learning_rate": [0.01, 0.03, 0.05, 0.1],
        f"{prefix}max_depth": [2, 3, 4, 5],
        f"{prefix}min_samples_leaf": [5, 10, 20],
        f"{prefix}l2_regularization": [0.0, 0.1, 1.0, 10.0],
    }


def train_regressor(
    X,
    y,
    normalize_embeddings: bool = False,
    random_search: bool = False,
    random_search_iters: int = 25,
    random_search_cv: int = 10,
    random_search_seed: int = 42,
):
    """
    Train a regression model.

    Args:
        X: Policy embeddings (n_samples, n_features)
        y: Rewards (n_samples,)

    Returns:
        Trained regression model
    """
    model, prefix = _make_hgbr_pipeline(normalize_embeddings)
    if random_search:
        param_dist = _hgbr_param_distributions(prefix)
        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=random_search_iters,
            cv=random_search_cv,
            scoring="neg_mean_absolute_error",
            random_state=random_search_seed,
            n_jobs=-1,
        )
        search.fit(X, y)
        print(f"✓ Random search best params: {search.best_params_}")
        return search.best_estimator_
    model.fit(X, y)
    return model


def save_model(model, model_path: str = "models/reward_regressor.pkl"):
    """
    Save the trained regression model to disk.

    Args:
        model: Trained LinearRegression model
        model_path: Path to save the model file
    """
    # Create models directory if it doesn't exist
    os.makedirs(
        os.path.dirname(model_path) if os.path.dirname(model_path) else ".",
        exist_ok=True,
    )

    # Save model using joblib
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")


def load_model(model_path: str = "models/reward_regressor.pkl"):
    """
    Load a trained regression model from disk.

    Args:
        model_path: Path to the saved model file

    Returns:
        Trained LinearRegression model

    Raises:
        FileNotFoundError: If the model file doesn't exist
    """
    if not os.path.exists(model_path):
        base_dir = os.path.dirname(model_path) or "."
        base_name = os.path.basename(model_path)
        candidates = []
        for spec in ("X1", "X5", "X10"):
            cand = os.path.join(base_dir, spec, base_name)
            if os.path.exists(cand):
                candidates.append(cand)
        extra_hint = ""
        if candidates:
            example = os.path.join(base_dir, "{spec}", base_name)
            extra_hint = (
                "\nModels were found in per-spec subfolders. "
                "Pass a spec-specific path or use the {spec} placeholder, e.g. "
                f"{example}."
            )
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "The regressor model hasn't been trained yet, or it was saved per-spec.\n"
            "Please run 'python pi2vec/train_regressor.py' first to train and save the model."
            f"{extra_hint}"
        )

    model = joblib.load(model_path)
    return model


def plot_regression_results(
    y_true, y_pred, save_path: str = "plots/regression_plot.pdf"
):
    """
    Plot regression results: predicted vs actual rewards.

    Args:
        y_true: True reward values
        y_pred: Predicted reward values
        save_path: Path to save the plot
    """
    # Create plots directory if it doesn't exist
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True
    )

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    # NMAE: Normalized Mean Absolute Error (MAE / mean of actual values)
    nmae = mae / np.mean(np.abs(y_true)) if np.mean(np.abs(y_true)) > 0 else np.inf

    # Create figure
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    # Scatter plot of predictions vs actual
    ax.scatter(
        y_true,
        y_pred,
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidth=0.5,
        label="Predictions",
    )

    # Perfect prediction line (y = x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )

    # Add regression line (actual model fit)
    # Fit a line through the predictions to show the trend
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    ax.plot(
        y_true,
        p(y_true),
        "b-",
        linewidth=2,
        alpha=0.8,
        label="Regression Line",
    )

    # Labels and title
    ax.set_xlabel("Actual Reward", fontsize=11)
    ax.set_ylabel("Predicted Reward", fontsize=11)
    ax.set_title("Performance Predictor: Policy Embedding → Reward", fontsize=12)

    # Add metrics text
    metrics_text = f"NMAE = {nmae:.3f}"
    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Legend
    ax.legend(loc="lower right", fontsize=9)

    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    # Also save as PNG
    png_path = save_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    plt.close()

    print(f"✓ Plot saved to {save_path}")
    print(f"✓ Plot saved to {png_path}")


def main(
    training_data_path: str = "data/regressor_training_data.json",
    model_path: str = "models/reward_regressor.pkl",
    plot_path: str = "plots/regression_plot.jpeg",
):
    """Main function to train regressor and generate plot."""
    # Load training data
    print("Loading training data...")
    X, y = load_training_data(json_path=training_data_path)

    # Check if we have any training data
    if len(X) == 0:
        raise ValueError(
            "No training data found. The training data file is empty.\n"
            "This means no policies were processed during successor model training.\n"
            "Please check that your data files exist and are in the correct location."
        )

    # Handle case where X might be 1D (shouldn't happen, but be safe)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")

    print("\n10-fold CV NMAE (90/10 splits):")
    kfold_nmae(X, y, k=10, normalize_embeddings=False)

    # Train model
    print("\nTraining HistGradientBoostingRegressor model...")
    model = train_regressor(X, y)
    print("✓ Model trained")

    # Save model
    print("\nSaving model...")
    save_model(model, model_path=model_path)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate and print metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    nmae = mae / np.mean(np.abs(y)) if np.mean(np.abs(y)) > 0 else np.inf

    print("\nModel Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  NMAE: {nmae:.4f}")

    # Create and save plot
    print("\nGenerating regression plot...")
    plot_regression_results(y, y_pred, save_path=plot_path)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
