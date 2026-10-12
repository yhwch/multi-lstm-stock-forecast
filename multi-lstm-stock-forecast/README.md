# MultiRNN: one LSTM per feature for time-series forecasting

A Python class that trains a separate LSTM model for every column of a multivariate time series, then predicts the next step for all columns at once. Demonstrated on Google stock prices (Open, High, Low, Close, Volume).

Built as a school assignment in the course *Maskininlärning och djupinlärning* (Machine Learning and Deep Learning).

## What it does

- Validates that the data is ready (numeric, no missing values, matching columns)
- Splits the dataset into one series per column while keeping the original date index
- Scales each column and builds sliding-window sequences
- Trains one LSTM per column, with per-column hyperparameters (units, activation, optimizer, epochs) and EarlyStopping
- Stores losses and predictions per column
- `predict()` returns next-step predictions for every column in one DataFrame
- Plots training vs validation loss and predictions vs actual values per column

## Example

```python
from multi_rnn import MultiRNN

rnn = MultiRNN(train=train_df, test=test_df, length=5, epochs=3, batch_size=1)
rnn.plot_predict_against_test_dataset_per_column(column="Close", figure_width=10, figure_height=5)
```

## Run it

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
python multi_rnn_test.py
```

`multi_rnn_test.py` loads and cleans the CSVs (dates, thousands separators), trains the models, and saves the plots.

## Tech

Python, TensorFlow/Keras (LSTM), scikit-learn, pandas, matplotlib
