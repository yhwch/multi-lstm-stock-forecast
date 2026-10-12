"""Test script for MultiRNN.

Loads the provided Google stock CSV files, cleans them into numeric DataFrames,
trains one LSTM per column using MultiRNN, prints a quick summary, saves a couple
of plots, and runs a simple next-step prediction.
"""

import pandas as pd

from multi_rnn import MultiRNN


def load_and_clean_stock_csv(path: str) -> pd.DataFrame:
    """Load Google stock CSV and return a clean numeric DataFrame indexed by Date.

    The assignment expects the developer to provide clean data:
    - Date parsed to datetime and used as index
    - No NaNs
    - No object dtypes (e.g., Volume like "7,380,500")
    """
    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError(f"Expected a 'Date' column in {path}")

    # Parse Date and set as index
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df = df.set_index("Date")

    # Remove thousands separators and convert everything to numeric
    # (This fixes columns like Volume that come in as object because of commas.)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Final sanity checks
    if df.isna().any().any():
        raise ValueError(f"NaNs detected after cleaning {path}")

    return df


def main():
    # Load and clean the training and test CSV files
    train = load_and_clean_stock_csv("Google_Stock_Price_Train.csv")
    test = load_and_clean_stock_csv("Google_Stock_Price_Test.csv")

    # Keep the exact same column order (important for MultiRNN validation)
    test = test[train.columns]

    # Create the MultiRNN instance with small epochs for a quick run
    rnn = MultiRNN(
        train=train,
        test=test,
        length=5,
        epochs=3,
        batch_size=1,
    )

    print("Columns/models trained:", list(rnn.models_per_column.keys()))

    first_col = list(rnn.models_per_column.keys())[0]
    print("\nFirst column:", first_col)
    print("Losses head:\n", rnn.models_per_column[first_col]["losses"].head())
    print("Predictions head:\n", rnn.models_per_column[first_col]["y_pred"].head())

    # Generate plots for all columns
    for col in rnn.models_per_column.keys():
        rnn.plot_loss_val_loss_per_column(
            column=col,
            figure_width=10,
            figure_height=5,
        )

        rnn.plot_predict_against_test_dataset_per_column(
            column=col,
            figure_width=10,
            figure_height=5,
        )

    # Example: predict the next timestep value for all columns using the last `length` rows
    print("\nNext-step prediction from last window:\n", rnn.predict(train.tail(rnn.length)))


if __name__ == "__main__":
    main()