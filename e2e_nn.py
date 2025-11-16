import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from math import sqrt
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K
from datetime import datetime
import optuna
from optuna.integration import TFKerasPruningCallback

DATA_PATH = "~/магистър ИИБФ/дипломна/data/model/processed/daily_sales_15875.csv"
BARCODE = "15875"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SEQ_LEN = 7
TEST_RATIO = 0.2
VALIDATION_FRAC = 0.15
SCALER_TYPE = "minmax"

LSTM_UNITS = 256
SECOND_LSTM_UNITS = 128
THIRD_LSTM_UNITS = 0
DROPOUT = 0.3
SECOND_DROPOUT = 0.3
THIRD_DROPOUT = 0.2
DENSE_UNITS = 0
LEARNING_RATE = 0.0005
LOSS_NAME = "mse"
EARLY_STOP_PATIENCE = 15
EPOCHS = 100
BATCH_SIZE = 64

RUN_OPTUNA_SEARCH = False
N_TRIALS = 20
TUNING_EPOCHS = 50

USE_LOG_TARGET = False
TARGET_TRANSFORM = "none"

COLS_TO_LOAD = ["Date", "Sales_QTTY"]
df = pd.read_csv(DATA_PATH, usecols=COLS_TO_LOAD, parse_dates=["Date"])

df = df.set_index("Date").sort_index()
full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
df = df.reindex(full_range)
df.index.name = "Date"
df['Sales_QTTY'].fillna(0, inplace=True)

df['target'] = df['Sales_QTTY'].astype(float)
FEATURE_COLS = ['target']

split_idx = int(len(df) * (1 - TEST_RATIO))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

scaler_cls = RobustScaler if SCALER_TYPE == 'robust' else MinMaxScaler
scaler = scaler_cls()

X_train_full = scaler.fit_transform(train_df[FEATURE_COLS])
X_test_full = scaler.transform(test_df[FEATURE_COLS])

y_train_full = X_train_full[:, 0]
y_test_full = X_test_full[:, 0]

def build_sequences(X_array, y_array, seq_len):
    X_list, y_list = [], []
    for i in range(seq_len, len(X_array)):
        X_list.append(X_array[i-seq_len:i, :])
        y_list.append(y_array[i])
    return np.array(X_list), np.array(y_list)

X_train, y_train = build_sequences(X_train_full, y_train_full, SEQ_LEN)
X_test, y_test = build_sequences(X_test_full, y_test_full, SEQ_LEN)
print(f"Train seq shape: {X_train.shape}; Test seq shape: {X_test.shape}")

val_split = int(len(X_train) * (1 - VALIDATION_FRAC))
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]
print(f"Initial split -> Train: {X_tr.shape}, Validation: {X_val.shape}")

raw_test_qty = test_df['Sales_QTTY'].values[SEQ_LEN:] # Подравняваме с y_test
y_true_qty_for_naive = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

naive_preds = y_true_qty_for_naive[:-1]
naive_truth = y_true_qty_for_naive[1:]
naive_rmse = sqrt(mean_squared_error(naive_truth, naive_preds))
naive_mae = mean_absolute_error(naive_truth, naive_preds)

print(f"Baseline RMSE - Naive: {naive_rmse:.2f}")
print(f"Baseline MAE - Naive: {naive_mae:.2f}")

def resolve_loss(name):
    if name == 'huber':
        return Huber()
    if name == 'mae':
        return 'mae'
    if name == 'mse':
        return 'mse'
    return Huber()

def build_lstm_model(lstm_units=LSTM_UNITS, second_lstm_units=SECOND_LSTM_UNITS, third_lstm_units=THIRD_LSTM_UNITS,
                       dropout=DROPOUT, second_dropout=SECOND_DROPOUT, third_dropout=THIRD_DROPOUT, 
                       dense_units=DENSE_UNITS, learning_rate=LEARNING_RATE,
                       loss_name=LOSS_NAME, **kwargs):
    K.clear_session()

    n_features = 1 
    
    layers = [Input(shape=(SEQ_LEN, n_features))]

    return_seq_1 = (second_lstm_units > 0)
    layers.append(LSTM(lstm_units, activation='tanh', return_sequences=return_seq_1))
    layers.append(Dropout(0.3))
    if second_lstm_units > 0:
        return_seq_2 = (third_lstm_units > 0)
        layers.append(LSTM(second_lstm_units, activation='tanh', return_sequences=return_seq_2))
            
    layers.append(Dropout(0.3))
    if third_lstm_units > 0:
        layers.append(LSTM(third_lstm_units, activation='tanh'))

    layers.append(Dense(1))
        
    m = Sequential(layers)
    m.compile(optimizer=Adam(learning_rate=learning_rate), loss=resolve_loss(loss_name))
    
    return m

def objective(trial):
    K.clear_session()
    
    n_features = 1
    
    learning_rate = trial.suggest_float('LEARNING_RATE', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('BATCH_SIZE', [32, 64, 128])
    loss_name = trial.suggest_categorical('LOSS_NAME', ['huber', 'mae', 'mse'])
    early_stop_patience = trial.suggest_int('EARLY_STOP_PATIENCE', 10, 20)
    
    num_lstm_layers = trial.suggest_int('NUM_LSTM_LAYERS', 1, 3)
    dense_units = trial.suggest_categorical('DENSE_UNITS', [16, 32, 64])
    
    layers = [Input(shape=(SEQ_LEN, n_features))]

    for i in range(num_lstm_layers):
        units_name = ['LSTM_UNITS', 'SECOND_LSTM_UNITS', 'THIRD_LSTM_UNITS'][i]
        dropout_name = ['DROPOUT', 'SECOND_DROPOUT', 'THIRD_DROPOUT'][i]

        units = trial.suggest_categorical(units_name, [64, 128, 256])
        dropout = trial.suggest_float(dropout_name, 0.1, 0.5)
        
        return_sequences = (i < num_lstm_layers - 1)
        
        layers.append(LSTM(
            units, activation='tanh', return_sequences=return_sequences, 
            dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(0.005)
        ))
    
    layers.append(Dense(dense_units, activation='relu'))
    layers.append(Dense(1))
    
    model = Sequential(layers)
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss=resolve_loss(loss_name))

    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=early_stop_patience, 
        restore_best_weights=True, 
        verbose=0
    )
    pruning_callback = TFKerasPruningCallback(trial, 'val_loss')

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=TUNING_EPOCHS,
        batch_size=batch_size,
        callbacks=[early_stop, pruning_callback],
        verbose=0
    )

    val_loss = model.evaluate(X_val, y_val, verbose=0)
    return val_loss


if RUN_OPTUNA_SEARCH:
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    best_trial = study.best_trial
    best_params = best_trial.params
    best_val_loss = best_trial.value
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(best_params)
    
    optuna_log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'barcode': BARCODE,
        'score_val_loss': best_val_loss,
        'model_type': 'E2E_LSTM'
    }
    optuna_log_entry.update(best_params) 

    log_path = "~/магистър ИИБФ/дипломна/data/model/processed/optuna_search_log_entries_е2е.csv"
    optuna_log_entry_df = pd.DataFrame([optuna_log_entry])
    
    if os.path.exists(log_path):
        optuna_log_entry_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        optuna_log_entry_df.to_csv(log_path, mode='w', header=True, index=False)
        

    sys.exit("Optuna Search end")

else:
    model = build_lstm_model(
        lstm_units=LSTM_UNITS, 
        second_lstm_units=SECOND_LSTM_UNITS, 
        third_lstm_units=THIRD_LSTM_UNITS,
        dropout=DROPOUT, 
        second_dropout=SECOND_DROPOUT, 
        third_dropout=THIRD_DROPOUT,
        dense_units=DENSE_UNITS, 
        learning_rate=LEARNING_RATE,
        loss_name=LOSS_NAME
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1)
    
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )


y_pred_target = model.predict(X_test)
y_true_target = y_test.reshape(-1, 1)

y_pred_qty = scaler.inverse_transform(y_pred_target)
y_true_qty = scaler.inverse_transform(y_true_target)

y_pred_qty = y_pred_qty.flatten()
y_true_qty = y_true_qty.flatten()

rmse_qty = sqrt(mean_squared_error(y_true_qty, y_pred_qty))
mae_qty = mean_absolute_error(y_true_qty, y_pred_qty)
mase_qty = mae_qty / naive_mae

print(f"LSTM Test RMSE: {rmse_qty:.2f}")
print(f"MAE: {mae_qty:.2f} бройки")
print(f"MASE: {mase_qty:.3f} (Naive MAE: {naive_mae:.2f})")

test_dates = test_df.index[SEQ_LEN:]
plt.figure(figsize=(12,5))
plt.plot(test_dates, y_true_qty, label="Actual", color='blue', alpha=0.7)
plt.plot(test_dates, y_pred_qty, label="E2E LSTM", color='red', linestyle='--')
plt.title(f"E2E LSTM (RMSE={rmse_qty:.2f}, MASE={mase_qty:.3f}) - {BARCODE}")
plt.xlabel("Дата")
plt.ylabel("Продажби")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = f'~/магистър ИИБФ/дипломна/data/plots/actual_vs_predicted_E2E_LSTM_{BARCODE}_{mase_qty:.3f}.svg'
plt.savefig(save_path, format='svg', bbox_inches='tight')
plt.show()

last_seq_scaled = X_train_full[-SEQ_LEN:]
last_block = last_seq_scaled.reshape(1, SEQ_LEN, 1)

next_target_scaled = model.predict(last_block)
next_qty = scaler.inverse_transform(next_target_scaled)[0,0]

print(f"Next-day forecast (barcode {BARCODE}): {max(0, round(next_qty))} units")

log_path = "~/магистър ИИБФ/дипломна/data/model/processed/lstm_log_entries_E2E.csv"

log_entry = {
    'timestamp': datetime.utcnow().isoformat(),
    'barcode': BARCODE,
    'model_type': 'E2E_LSTM',
    'seq_len': SEQ_LEN,
    'test_ratio': TEST_RATIO,
    'validation_frac': VALIDATION_FRAC,
    'scaler_type': SCALER_TYPE,
    'lstm_units': LSTM_UNITS,
    'second_lstm_units': SECOND_LSTM_UNITS,
    'third_lstm_units': THIRD_LSTM_UNITS,
    'dropout': DROPOUT,
    'second_dropout': SECOND_DROPOUT,
    'third_dropout': THIRD_DROPOUT,
    'dense_units': DENSE_UNITS,
    'learning_rate': LEARNING_RATE,
    'loss_name': LOSS_NAME,
    'epochs_trained': len(history.history['loss']),
    'batch_size': BATCH_SIZE,
    'use_log_target': USE_LOG_TARGET,
    'target_transform': TARGET_TRANSFORM,
    'naive_rmse': round(naive_rmse, 4),
    'ma7_rmse': 0,
    'lstm_rmse': round(rmse_qty, 4),
    'naive_mae': round(naive_mae, 4),
    'lstm_mae': round(mae_qty, 4),
    'lstm_mase': round(mase_qty, 4),
    'next_day_forecast': max(0, round(next_qty)),
    'train_sequences': X_train.shape[0],
    'test_sequences': X_test.shape[0],
    'feature_count': 1
}
log_entry_df = pd.DataFrame([log_entry])

if os.path.exists(log_path):
    log_entry_df.to_csv(log_path, mode='a', header=False, index=False)
else:
    log_entry_df.to_csv(log_path, mode='w', header=True, index=False)

