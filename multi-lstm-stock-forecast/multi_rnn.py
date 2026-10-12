# multi_rnn.py
"""MultiRNN – Deep Learning Inlämningsuppgift #2

This file implements a MultiRNN class that trains one LSTM model per feature (column)
for a multivariate time-series dataset.

Main idea (from the assignment):
- Split the multifeature dataset into single-column datasets (keeping the original index)
- Train one LSTM per column
- Store losses and predictions per column
- Provide plotting helpers and a single predict() that returns predictions for all columns

Note: The developer/user is responsible for providing clean data (numeric, no NaNs,
no object dtypes). This class validates the input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


class MultiRNN:
    """Build one LSTM model per column for multifeature time-series datasets."""

    def __init__(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        length: int,
        LSTM_units: Union[int, List[int]] = 50,
        activation: Union[str, List[str]] = "tanh",
        optimizer: Union[str, List[str]] = "adam",
        batch_size: int = 1,
        epochs: Union[int, List[int]] = 25,
    ):
        # Basic input validation
        if not isinstance(train, pd.DataFrame):
            raise ValueError("train must be a pandas DataFrame")
        if not isinstance(test, pd.DataFrame):
            raise ValueError("test must be a pandas DataFrame")
        if not isinstance(length, int) or length <= 0:
            raise ValueError("length must be a positive integer")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self.length = length
        self.batch_size = batch_size

        # Assignment requirement: check datasets are ready
        self.train = self.ready_for_processing(train)
        self.test = self.ready_for_processing(test)

        # Train and test must match in columns/order
        if list(self.train.columns) != list(self.test.columns):
            raise ValueError(
                "train and test must have the same columns in the same order. "
                f"train={list(self.train.columns)} test={list(self.test.columns)}"
            )

        self.columns: List[str] = list(self.train.columns)
        self.n_features: int = len(self.columns)

        # Normalize hyperparameters so we always have a per-column list
        self.LSTM_units = self._normalize_hyper(LSTM_units, "LSTM_units", expected_type=int, positive_int=True)
        self.activation = self._normalize_hyper(activation, "activation", expected_type=str)
        self.optimizer = self._normalize_hyper(optimizer, "optimizer", expected_type=str)
        self.epochs = self._normalize_hyper(epochs, "epochs", expected_type=int, positive_int=True)

        # Dict-of-dicts (required by assignment)
        self.models_per_column: Dict[str, Dict[str, Any]] = {}

        # Tip in the PDF: constructor should build models
        self.models_per_column = self.generate_model_list_per_column_in_data_set(
            train=self.train,
            test=self.test,
            length=self.length,
            LSTM_units=self.LSTM_units,
            activation=self.activation,
            optimizer=self.optimizer,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

    # ----------------------------
    # Validation + helpers
    # ----------------------------
    def _normalize_hyper(
        self,
        value: Union[Any, List[Any]],
        name: str,
        expected_type: type,
        positive_int: bool = False,
    ) -> List[Any]:
        """Allow a scalar hyperparameter or a list with one value per feature."""
        if isinstance(value, list):
            if len(value) != self.n_features:
                raise ValueError(
                    f"{name} must be a scalar or a list with length == number of columns ({self.n_features}). "
                    f"Got {len(value)}"
                )
            for i, v in enumerate(value):
                if not isinstance(v, expected_type):
                    raise ValueError(f"{name}[{i}] must be of type {expected_type.__name__}")
                if positive_int and (not isinstance(v, int) or v <= 0):
                    raise ValueError(f"{name}[{i}] must be a positive integer")
            return value

        # Scalar value -> repeat for each column
        if not isinstance(value, expected_type):
            raise ValueError(f"{name} must be of type {expected_type.__name__} or list[{expected_type.__name__}]")
        if positive_int and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
        return [value for _ in range(self.n_features)]

    def ready_for_processing(self, data_set: pd.DataFrame) -> pd.DataFrame:
        """Check if dataset is ready for processing (no NaNs, only numeric columns)."""
        if not isinstance(data_set, pd.DataFrame):
            raise ValueError("data_set must be a pandas DataFrame")

        if data_set.isna().any().any():
            nan_cols = data_set.columns[data_set.isna().any()].tolist()
            raise ValueError(f"Dataset contains NaN values in columns: {nan_cols}")

        bad_cols = data_set.select_dtypes(include=["object", "category"]).columns.tolist()
        if bad_cols:
            raise ValueError(
                "Dataset contains non-numeric columns (object/category). Clean them first. "
                f"Columns: {bad_cols}"
            )

        non_numeric = [c for c in data_set.columns if not pd.api.types.is_numeric_dtype(data_set[c])]
        if non_numeric:
            raise ValueError(f"Dataset contains non-numeric columns: {non_numeric}")

        return data_set.copy()

    def generate_data_set_per_column_with_original_index(self, data_set: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate single-column datasets that keep the original index."""
        data_set = self.ready_for_processing(data_set)
        return {col: data_set[[col]].copy() for col in data_set.columns}

    # ----------------------------
    # Model building per column
    # ----------------------------
    def _make_sequences(self, arr_2d: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create (X, y) sequences from an array shaped (n, 1)."""
        X, y = [], []
        for i in range(length, len(arr_2d)):
            X.append(arr_2d[i - length : i, 0])
            y.append(arr_2d[i, 0])
        X = np.array(X, dtype=float).reshape(-1, length, 1)
        y = np.array(y, dtype=float)
        return X, y

    def build_model_per_column(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        length: int,
        LSTM_units: int,
        activation: str,
        optimizer: str,
        batch_size: int,
        epochs: int,
    ) -> Tuple[Any, pd.DataFrame, str, MinMaxScaler, pd.DataFrame, pd.DataFrame]:
        """Train a single LSTM model for one column and return artifacts for plotting."""
        if train.shape[1] != 1 or test.shape[1] != 1:
            raise ValueError("build_model_per_column expects single-column train/test DataFrames")

        col_name = train.columns[0]
        model_name = f"LSTM_{col_name}"

        # Scale per-column (fit only on train)
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_vals = train.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)
        test_vals = test.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)
        train_scaled = scaler.fit_transform(train_vals)
        test_scaled = scaler.transform(test_vals)

        # Build sequences
        X_train, y_train = self._make_sequences(train_scaled, length)
        X_test, y_test = self._make_sequences(test_scaled, length)

        # LSTM model (simple on purpose)
        model = Sequential([
            Input(shape=(length, 1)),
            LSTM(int(LSTM_units), activation=str(activation)),
            Dense(1),
        ])
        model.compile(optimizer=str(optimizer), loss="mse")

        # Early stopping as requested (patience=2)
        early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=int(epochs),
            batch_size=int(batch_size),
            verbose=0,
            callbacks=[early_stop],
        )

        losses_df = pd.DataFrame({
            "loss": history.history.get("loss", []),
            "val_loss": history.history.get("val_loss", []),
        })

        # Predict on test and invert scaling
        y_pred_scaled = model(X_test, training=False).numpy()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
        y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)

        # Align with test index (sequence window drops the first `length` rows)
        aligned_index = test.index[length:]
        y_test_df = pd.DataFrame({col_name: y_true}, index=aligned_index)
        y_pred_df = pd.DataFrame({col_name: y_pred}, index=aligned_index)

        return model, losses_df, model_name, scaler, y_test_df, y_pred_df

    def generate_model_list_per_column_in_data_set(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        length: int,
        LSTM_units: List[int],
        activation: List[str],
        optimizer: List[str],
        batch_size: int,
        epochs: List[int],
    ) -> Dict[str, Dict[str, Any]]:
        """Train models for all columns and return a dict-of-dicts."""
        train_dict = self.generate_data_set_per_column_with_original_index(train)
        test_dict = self.generate_data_set_per_column_with_original_index(test)

        out: Dict[str, Dict[str, Any]] = {}

        for i, col in enumerate(self.columns):
            model, losses, model_name, scaler, y_test_df, y_pred_df = self.build_model_per_column(
                train=train_dict[col],
                test=test_dict[col],
                length=length,
                LSTM_units=int(LSTM_units[i]),
                activation=str(activation[i]),
                optimizer=str(optimizer[i]),
                batch_size=int(batch_size),
                epochs=int(epochs[i]),
            )

            out[col] = {
                "model": model,
                "losses": losses,
                "model_name": model_name,
                # Extra artifacts are useful for plotting/predict()
                "scaler": scaler,
                "y_test": y_test_df,
                "y_pred": y_pred_df,
                "length": length,
            }

        return out

    # ----------------------------
    # Prediction + plotting
    # ----------------------------
    def predict(self, data_row: pd.DataFrame) -> pd.DataFrame:
        """Predict the next value for ALL columns using the last `length` timesteps."""
        if not isinstance(data_row, pd.DataFrame):
            raise ValueError("data_row must be a pandas DataFrame")

        missing = [c for c in self.columns if c not in data_row.columns]
        extra = [c for c in data_row.columns if c not in self.columns]
        if missing:
            raise ValueError(f"data_row is missing required columns: {missing}")
        if extra:
            raise ValueError(f"data_row has extra columns not in training data: {extra}")

        if len(data_row) < self.length:
            raise ValueError(f"data_row must have at least {self.length} rows to build a sequence window")

        preds: Dict[str, float] = {}

        for col in self.columns:
            artifacts = self.models_per_column[col]
            model = artifacts["model"]
            scaler: MinMaxScaler = artifacts["scaler"]

            window = data_row[col].to_numpy(dtype=float)[-self.length :].reshape(-1, 1)
            window_scaled = scaler.transform(window).reshape(1, self.length, 1)

            pred_scaled = model(window_scaled, training=False).numpy()
            pred = scaler.inverse_transform(pred_scaled).reshape(-1)[0]
            preds[col] = float(pred)

        return pd.DataFrame([preds], columns=self.columns)

    def plot_predict_against_test_dataset_per_column(
        self,
        column: str,
        figure_width: int,
        figure_height: int,
        save_plot_name: str = "plot_test_vs_predict_",
    ) -> None:
        """Plot test vs predicted values for one column and save the plot."""
        if column not in self.models_per_column:
            raise ValueError(f"Unknown column: {column}. Available: {list(self.models_per_column.keys())}")

        y_test_df: pd.DataFrame = self.models_per_column[column]["y_test"]
        y_pred_df: pd.DataFrame = self.models_per_column[column]["y_pred"]

        plt.figure(figsize=(figure_width, figure_height))
        plt.plot(y_test_df.index, y_test_df[column].values, label="Test (Actual)")
        plt.plot(y_pred_df.index, y_pred_df[column].values, label="Predicted")
        plt.title(f"Test vs Predicted – {column}")
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_plot_name}{column}.png", dpi=150)
        plt.close()

    def plot_loss_val_loss_per_column(
        self,
        column: str,
        figure_width: int,
        figure_height: int,
        save_plot_name: str = "Plot_Loss_val_loss_",
    ) -> None:
        """Plot loss vs val_loss for one column and save the plot."""
        if column not in self.models_per_column:
            raise ValueError(f"Unknown column: {column}. Available: {list(self.models_per_column.keys())}")

        losses: pd.DataFrame = self.models_per_column[column]["losses"]

        plt.figure(figsize=(figure_width, figure_height))
        plt.plot(losses.index, losses["loss"].values, label="loss")
        plt.plot(losses.index, losses["val_loss"].values, label="val_loss")
        plt.title(f"Loss vs Val Loss – {column}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_plot_name}{column}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    # Minimal demo only (loading/cleaning data is intentionally outside the class).
    pass