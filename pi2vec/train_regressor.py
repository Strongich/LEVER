"""
Train a linear regression model to predict reward from policy embeddings.
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


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


def train_regressor(X, y):
    """
    Train a linear regression model.

    Args:
        X: Policy embeddings (n_samples, n_features)
        y: Rewards (n_samples,)

    Returns:
        Trained LinearRegression model
    """
    model = LinearRegression()
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
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "The regressor model hasn't been trained yet.\n"
            "Please run 'python pi2vec/train_regressor.py' first to train and save the model."
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
    r2 = r2_score(y_true, y_pred)
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
    metrics_text = f"R² = {r2:.3f}\nNMAE = {nmae:.3f}"
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

    # Train model
    print("\nTraining linear regression model...")
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
