import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
import sys
from math import sqrt
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from datetime import datetime
import optuna
from optuna.integration import TFKerasPruningCallback    
import math
import matplotlib.dates as mdates


DATA_PATH = "~/магистър ИИБФ/дипломна/data/model/processed/daily_sales_2000008_with_weather.csv"
BARCODE = "2000008" #2033 2000008 142 1498
RANDOM_SEED = 42

SEQ_LEN = 7
TEST_RATIO = 0.2
VALIDATION_FRAC = 0.15
SCALER_TYPE = "robust"  # robust | minmax

# hyperparameters
LSTM_UNITS = 256
SECOND_LSTM_UNITS = 128
THIRD_LSTM_UNITS = 64
DROPOUT = 0.3
SECOND_DROPOUT = 0.5
THIRD_DROPOUT = 0.5
DENSE_UNITS = 64
LEARNING_RATE = 0.001
LOSS_NAME = "huber"     # huber | mae | mse
EARLY_STOP_PATIENCE = 20
EPOCHS = 300
BATCH_SIZE = 64

USE_LOG_TARGET = True
TARGET_TRANSFORM = "log1p"  # log1p | none

# feature flags
INCLUDE_LAG2 = True
INCLUDE_LAG3 = True
INCLUDE_LAG4 = False
INCLUDE_LAG14 = True
INCLUDE_LAG28 = False
INCLUDE_ROLLING_14 = True
INCLUDE_ROLLING_MAX_7 = True
INCLUDE_ROLLING_MIN_7 = True
INCLUDE_SPIKE_FEATURES = True
SPIKE_QUANTILE = 0.90
OVERSAMPLE_SPIKES = True
OVERSAMPLE_FACTOR = 10

INCLUDE_WEATHER = True
WEATHER_COLS = ["max_temp", "precipitation", "sunshine_hours"]

RUN_OPTUNA_SEARCH = False
N_TRIALS = 20
TUNING_EPOCHS = 50

np.random.seed(RANDOM_SEED)

COLS_TO_LOAD = ["Date", "Sales_QTTY", "Delivery_QTTY"] + (WEATHER_COLS if INCLUDE_WEATHER else [])
df = pd.read_csv(DATA_PATH, usecols=COLS_TO_LOAD, parse_dates=["Date"])
df.describe()

df['Sales_QTTY'].plot()

df = df.set_index("Date").sort_index()
full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
df = df.reindex(full_range)
df.index.name = "Date"
df['Sales_QTTY'].fillna(0, inplace=True)
df['Delivery_QTTY'] = df['Delivery_QTTY'].fillna(0)

# features
FEATURE_COLS = [
    'weekday_sin', 'weekday_cos', 
    'lag_1', 'lag_7',                                  
    'rolling_mean_7', 'rolling_std_7',                  
    'day_of_month_sin', 'day_of_month_cos',
    'is_holiday_exact_day',
    'is_holiday_flag',
    'is_pre_holiday',
    'is_pre_pre_holiday', 'stock',
]

df['weekday'] = df.index.weekday
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

df['lag_1'] = df['Sales_QTTY'].shift(1)
df['lag_7'] = df['Sales_QTTY'].shift(7)

df['rolling_mean_7'] = df['Sales_QTTY'].rolling(7).mean().shift(1)
df['rolling_std_7'] = df['Sales_QTTY'].rolling(7).std(ddof=0).shift(1)

df['day_of_month'] = df.index.day
df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)


count_zeros = len(df[(df['Delivery_QTTY'] == 0) & (df['weekday'] != 6)])
print(f"Брой недели с 0 доставки: {count_zeros}")

# 1. Създаваме филтър за проблемните редове
# Условие: Доставка е 0 И Денят НЕ е Неделя (6)
mask = (df['Delivery_QTTY'] == 0) & (df['weekday'] != 6)
print(f"Открити са {mask.sum()} липсващи доставки, които ще бъдат попълнени.")
df.loc[mask, 'Delivery_QTTY'] = df.loc[mask, 'Sales_QTTY']

df['stock_change'] = df['Delivery_QTTY'] - df['Sales_QTTY']

recalculated_stock = []
current_stock = 0

for delivery, sales in zip(df['Delivery_QTTY'], df['Sales_QTTY']):
    # Доставката добавя към наличния stock
    current_stock += delivery
    # Продажбите намаляват stock, но не може да стане под 0
    current_stock = max(0, current_stock - sales)
    recalculated_stock.append(current_stock)

df['stock'] = recalculated_stock
df['Delivery_QTTY'].sum(), df['Sales_QTTY'].sum()


df[["Sales_QTTY", "Delivery_QTTY", "stock_change", "stock"]].to_csv("~/магистър ИИБФ/дипломна/data/model/processed/40sto.csv")

print("--- ПРОВЕРКА НА ПРЕХОДА 2024 -> 2025 ---")
# Нека видим края на 2024 и началото на 2025, за да сме сигурни, че е сработило
# Намерете индекса, където свършва 2024 (приблизително)
print(df[['Delivery_QTTY', 'Sales_QTTY', 'stock']].loc['2024-12-25':'2025-01-15'])

bg_holidays = holidays.BG(years=df.index.year.unique())
official_dates = [date.strftime('%Y-%m-%d') for date in bg_holidays.keys()]

custom_holidays = pd.read_csv("~/магистър ИИБФ/дипломна/data/model/processed/custom_holidays.csv", usecols=["Date"], parse_dates=["Date"])
custom_holidays_list = custom_holidays['Date'].dt.strftime('%Y-%m-%d').tolist()
print(custom_holidays_list)

all_holiday_dates_str = list(set(official_dates + custom_holidays_list))
all_holiday_dates = pd.to_datetime(all_holiday_dates_str)

df['is_holiday_exact_day'] = df.index.isin(all_holiday_dates).astype(int)
df['is_holiday_flag'] = df['is_holiday_exact_day'].rolling(window=7, center=True, min_periods=1).max().fillna(0).astype(int)

df['is_pre_holiday'] = df['is_holiday_exact_day'].shift(-1).fillna(0).astype(int)
df['is_pre_pre_holiday'] = df['is_holiday_exact_day'].shift(-2).fillna(0).astype(int)

if INCLUDE_SPIKE_FEATURES:
    spike_threshold = df['Sales_QTTY'].quantile(SPIKE_QUANTILE)
    df['spike_flag'] = (df['Sales_QTTY'] >= spike_threshold).astype(int)
    counter = 0; days_since_last_spike = []
    for q in df['Sales_QTTY']:
        if q >= spike_threshold:
            counter = 0
        else:
            counter += 1
        days_since_last_spike.append(counter)
    df['days_since_last_spike'] = days_since_last_spike
    df['last_spike_qtty'] = np.where(df['Sales_QTTY'] >= spike_threshold, df['Sales_QTTY'], 0)
    df['last_spike_qtty'] = df['last_spike_qtty'].replace(to_replace=0, method='ffill').shift(1).fillna(0)
    FEATURE_COLS.extend(['spike_flag', 'days_since_last_spike', 'last_spike_qtty'])

else:
    spike_threshold = None
    df['spike_flag'] = 0
    df['days_since_last_spike'] = 0
    
    

if INCLUDE_LAG2: 
    FEATURE_COLS.append('lag_2')
    df['lag_2'] = df['Sales_QTTY'].shift(2)    
if INCLUDE_LAG3: 
    FEATURE_COLS.append('lag_3')
    df['lag_3'] = df['Sales_QTTY'].shift(3)
if INCLUDE_LAG4: 
    FEATURE_COLS.append('lag_4')
    df['lag_4'] = df['Sales_QTTY'].shift(4)
if INCLUDE_LAG14: 
    FEATURE_COLS.append('lag_14')
    df['lag_14'] = df['Sales_QTTY'].shift(14)
if INCLUDE_LAG28: 
    FEATURE_COLS.append('lag_28')
    df['lag_28'] = df['Sales_QTTY'].shift(28)
    
if INCLUDE_ROLLING_MAX_7: 
    FEATURE_COLS.append('rolling_max_7')
    df['rolling_max_7'] = df['Sales_QTTY'].rolling(7).max().shift(1)
if INCLUDE_ROLLING_MIN_7: 
    FEATURE_COLS.append('rolling_min_7')
    df['rolling_min_7'] = df['Sales_QTTY'].rolling(7).min().shift(1)


if INCLUDE_ROLLING_14:
    FEATURE_COLS.extend(['rolling_mean_14', 'rolling_std_14'])
    df['rolling_mean_14'] = df['Sales_QTTY'].rolling(14).mean().shift(1)
    df['rolling_std_14'] = df['Sales_QTTY'].rolling(14).std(ddof=0).shift(1)

if INCLUDE_WEATHER:
    FEATURE_COLS.extend(WEATHER_COLS)

df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

# target
if USE_LOG_TARGET and TARGET_TRANSFORM == 'log1p':
    df['target'] = np.log1p(df['Sales_QTTY'])
else:
    df['target'] = df['Sales_QTTY'].astype(float)

df.info()

# split test/train
split_idx = int(len(df) * (1 - TEST_RATIO))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# seasonal decomposition weekly
decomposition_weekly = seasonal_decompose(df['Sales_QTTY'], model='additive', period=7)
    
fig = decomposition_weekly.plot()
fig.axes[0].set_title('Продажби', fontsize=12)
fig.axes[1].set_title('Тренд', fontsize=12)
fig.axes[2].set_title('Сезонност', fontsize=12)
fig.axes[3].set_title('Остатъци', fontsize=12)

fig.set_size_inches(14, 10)
plt.suptitle(f'Седмична сезонна декомпозиция (Продукт: {BARCODE})', y=1.02, fontsize=16)


save_path = f"~/магистър ИИБФ/дипломна/data/plots/seasonal_decomposition_{BARCODE}_weekly.svg"
fig.savefig(save_path, format='svg', bbox_inches='tight')
plt.show()


# seasonal decomposition yearly
decomposition_yearly = seasonal_decompose(df['Sales_QTTY'], model='additive', period=365)
    
fig = decomposition_yearly.plot()
fig.axes[0].set_title('Продажби', fontsize=12)
fig.axes[1].set_title('Тренд', fontsize=12)
fig.axes[2].set_title('Сезонност', fontsize=12)
fig.axes[3].set_title('Остатъци', fontsize=12)

fig.set_size_inches(14, 10)
plt.suptitle(f'Годишна сезонна декомпозиция (Продукт: {BARCODE})', y=1.02, fontsize=16)


save_path = f"~/магистър ИИБФ/дипломна/data/plots/seasonal_decomposition_{BARCODE}_yearly.svg"
fig.savefig(save_path, format='svg', bbox_inches='tight')
plt.show()


scaler_cls = RobustScaler if SCALER_TYPE == 'robust' else MinMaxScaler
x_scaler = scaler_cls()
X_train_full = x_scaler.fit_transform(train_df[FEATURE_COLS])
X_test_full = x_scaler.transform(test_df[FEATURE_COLS])

y_train_full = train_df['target'].values
y_test_full = test_df['target'].values

def build_sequences(X_array, y_array, seq_len):
    X_list, y_list = [], []
    for i in range(seq_len, len(X_array)):
        X_list.append(X_array[i-seq_len:i, :])
        y_list.append(y_array[i])
    return np.array(X_list), np.array(y_list)

X_train, y_train = build_sequences(X_train_full, y_train_full, SEQ_LEN)
X_test, y_test = build_sequences(X_test_full, y_test_full, SEQ_LEN)
print(f"Train seq shape: {X_train.shape}; Test seq shape: {X_test.shape}")

# split train/validation
val_split = int(len(X_train) * (1 - VALIDATION_FRAC))
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]
print(f"Initial split -> Train: {X_tr.shape}, Validation: {X_val.shape}")


if OVERSAMPLE_SPIKES and spike_threshold is not None:
    train_df_tr_only = train_df.iloc[:val_split]
    raw_train_qty_tr = train_df_tr_only['Sales_QTTY'].values
    
    aligned_qty_tr = raw_train_qty_tr[SEQ_LEN:] 
    
    if len(aligned_qty_tr) != len(y_tr):
         aligned_qty_tr = raw_train_qty_tr[SEQ_LEN:len(y_tr) + SEQ_LEN]

    high_idx = np.where(aligned_qty_tr >= spike_threshold)[0]
    
    if len(high_idx) > 0 and OVERSAMPLE_FACTOR > 0:
        print(f"Found {len(high_idx)} spike windows in the clean training set (X_tr).")
        spike_values = aligned_qty_tr[high_idx]
    
        print(f"Spike Threshold (Q{SPIKE_QUANTILE*100:.0f}): {spike_threshold:.2f} units")
        print(f"Actual spike quantities found: {np.sort(spike_values)}")
        
        X_high = X_tr[high_idx]; y_high = y_tr[high_idx]

        X_tr = np.concatenate([X_tr] + [X_high]*OVERSAMPLE_FACTOR, axis=0)
        y_tr = np.concatenate([y_tr] + [y_high]*OVERSAMPLE_FACTOR, axis=0)
        
        print(f"Oversampled training set (X_tr) shape: {X_tr.shape}")
    else:
        print("No spikes found for oversampling in the training subset or factor is 0.")



# baseline models
raw_train_target = train_df['Sales_QTTY'].values
raw_test_target = test_df['Sales_QTTY'].values

naive_preds = raw_test_target[:-1]; naive_truth = raw_test_target[1:]
naive_rmse = sqrt(mean_squared_error(naive_truth, naive_preds))
naive_mae = mean_absolute_error(naive_truth, naive_preds)

ma_series = pd.Series(np.concatenate([raw_train_target, raw_test_target]))
ma_pred = ma_series.rolling(7).mean().shift(1).iloc[len(raw_train_target):].values
valid_mask = ~np.isnan(ma_pred)
ma_rmse = sqrt(mean_squared_error(raw_test_target[valid_mask], ma_pred[valid_mask]))

print(f"Baseline RMSE - Naive: {naive_rmse:.2f}; 7d MA: {ma_rmse:.2f}")
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
        
    layers = [Input(shape=(SEQ_LEN, len(FEATURE_COLS)))]
    layers.append(LSTM(lstm_units, activation='tanh', return_sequences=True, dropout=dropout, recurrent_dropout=0.2, kernel_regularizer=l2(0.005)))
    
    if second_lstm_units > 0:
        if third_lstm_units > 0:
            layers.append(LSTM(second_lstm_units, activation='tanh', return_sequences=True, dropout=second_dropout, recurrent_dropout=0.2, kernel_regularizer=l2(0.005)))
        else:
            layers.append(LSTM(second_lstm_units, activation='tanh', dropout=second_dropout, recurrent_dropout=0.2, kernel_regularizer=l2(0.005)))
                        
    if third_lstm_units > 0:
        layers.append(LSTM(third_lstm_units, activation='tanh',dropout=third_dropout, recurrent_dropout=0.2, kernel_regularizer=l2(0.005)))
    elif second_lstm_units <= 0:
        pass

    layers.append(Dense(dense_units, activation='relu'))
    #layers.append(Dropout(0.2))
    layers.append(Dense(1))
        
    m = Sequential(layers)
    m.compile(optimizer=Adam(learning_rate=learning_rate), loss=resolve_loss(loss_name))
    
    return m


def objective(trial):
    K.clear_session()
    
    learning_rate = trial.suggest_float('LEARNING_RATE', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('BATCH_SIZE', [32, 64, 128])
    loss_name = trial.suggest_categorical('LOSS_NAME', ['huber', 'mae', 'mse'])
    early_stop_patience = trial.suggest_int('EARLY_STOP_PATIENCE', 10, 20)
    
    num_lstm_layers = trial.suggest_int('NUM_LSTM_LAYERS', 1, 3)
    dense_units = trial.suggest_categorical('DENSE_UNITS', [16, 32, 64])
    
    layers = [Input(shape=(SEQ_LEN, len(FEATURE_COLS)))]

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
    print("\n--- OPTUNA HYPERPARAMETER TUNING (TPE Sampler) ---")
    
    # 1. Създаване на Study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    
    # 2. Оптимизация
    print(f"Стартиране на {N_TRIALS} итерации (опита)...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # 3. Вземане на най-добрите параметри
    best_trial = study.best_trial
    best_params = best_trial.params
    best_val_loss = best_trial.value
    
    print("\n--- НАМЕРЕНИ ОПТИМАЛНИ ПАРАМЕТРИ (OPTUNA) ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(best_params)
    
    # 4. СЪЗДАВАНЕ НА ЛОГ ЗАПИС (както го искахте)
    optuna_log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'barcode': BARCODE,
        'score_val_loss': best_val_loss,
    }
    # Добавяме всички намерени параметри (LEARNING_RATE, BATCH_SIZE, LSTM_UNITS и т.н.)
    optuna_log_entry.update(best_params) 

    # 5. ЗАПИСВАНЕ НА ЛОГ ФАЙЛА
    log_path = "~/магистър ИИБФ/дипломна/data/model/processed/optuna_search_log_entries.csv"
    optuna_log_entry_df = pd.DataFrame([optuna_log_entry])
    
    if os.path.exists(log_path):
        optuna_log_entry_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        optuna_log_entry_df.to_csv(log_path, mode='w', header=True, index=False)
        
    print(f"Резултатите от Optuna са записани в: {log_path}")

    sys.exit("Optuna Search приключи. Спиране на изпълнението.")

else:
    print("\n--- СТАНДАРТНО ОБУЧЕНИЕ (Използват се ръчни хиперпараметри) ---")

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

model.summary()

# test
y_pred_target = model.predict(X_test).flatten()
y_true_target = y_test
if USE_LOG_TARGET and TARGET_TRANSFORM == 'log1p':
    y_pred_qty = np.expm1(y_pred_target)
    y_true_qty = np.expm1(y_true_target)
else:
    y_pred_qty = y_pred_target
    y_true_qty = y_true_target

rmse_qty = sqrt(mean_squared_error(y_true_qty, y_pred_qty))
rmse_target = sqrt(mean_squared_error(y_true_target, y_pred_target))
print(f"LSTM Test RMSE (quantity): {rmse_qty:.2f}")

print(f"LSTM Test RMSE: {rmse_target:.2f}")

mae_qty = mean_absolute_error(y_true_qty, y_pred_qty)
mase_qty = mae_qty / naive_mae
print(f"MAE: {mae_qty:.2f} бройки")
print(f"MASE: {mase_qty:.3f}")

ma_mae = mean_absolute_error(raw_test_target[valid_mask], ma_pred[valid_mask])
ma_mase = ma_mae / naive_mae


test_dates = test_df.index[SEQ_LEN:]
plt.figure(figsize=(12,5))
plt.plot(test_dates, y_true_qty, label='Actual', alpha=0.7)
plt.plot(test_dates, y_pred_qty, label='LSTM', linestyle='--')
plt.title(f'LSTM (RMSE={rmse_qty:.2f} MASE: {mase_qty:.3f}) - {BARCODE}')
plt.legend(); 
plt.grid(True); 
plt.tight_layout(); 
plt.savefig(f'~/магистър ИИБФ/дипломна/data/plots/actual_vs_predicted_LSTMwF_{BARCODE}_{mase_qty:.3f}.svg', format='svg', bbox_inches='tight')
plt.show()


errors = y_true_qty - y_pred_qty


plt.figure(figsize=(8,4))
sns.histplot(errors, kde=True)
plt.title(f"Хистограма на прогнозните грешки (PRODUCT: {BARCODE}, MAE: {mae_qty:.2f})")
plt.xlabel("Грешка")
plt.ylabel("Честота")
plt.show()

# forecast next day sales
last_block_df = df[FEATURE_COLS].tail(SEQ_LEN)
last_block = x_scaler.transform(last_block_df).reshape(1, SEQ_LEN, len(FEATURE_COLS))
next_target = model.predict(last_block)[0,0]
if USE_LOG_TARGET and TARGET_TRANSFORM == 'log1p':
    next_qty = max(0, round(np.expm1(next_target)))
else:
    next_qty = max(0, round(next_target))
    
print(f"Next day forecast (barcode {BARCODE}): {next_qty} units")

print(f"Summary RMSE -> Naive: {naive_rmse:.2f} | 7d MA: {ma_rmse:.2f} | LSTM: {rmse_qty:.2f}")
print(f"Summary MAE  -> Naive: {naive_mae:.2f} | 7d MA: {ma_mae:.2f} | LSTM: {mae_qty:.2f}")
print(f"Summary MASE -> Naive: 1.00 | 7d MA: {ma_mase:.2f} | LSTM: {mase_qty:.2f}")


log_path = "~/магистър ИИБФ/дипломна/data/model/processed/lstm_log_entries_Y.csv"

log_entry = {
    'timestamp': datetime.utcnow().isoformat(),
    'barcode': BARCODE,
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
    'include_lag2': INCLUDE_LAG2,
    'include_lag3': INCLUDE_LAG3,
    'include_lag4': INCLUDE_LAG4,
    'include_rolling_14': INCLUDE_ROLLING_14,
    'include_spike_features': INCLUDE_SPIKE_FEATURES,
    'spike_quantile': SPIKE_QUANTILE,
    'oversample_spikes': OVERSAMPLE_SPIKES,
    'oversample_factor': OVERSAMPLE_FACTOR,
    'naive_rmse': round(naive_rmse, 4),
    'ma7_rmse': round(ma_rmse, 4),
    'lstm_rmse': round(rmse_qty, 4),
    'naive_mae': round(naive_mae, 4),
    'lstm_mae': round(mae_qty, 4),
    'lstm_mase': round(mase_qty, 4),
    'next_day_forecast': next_qty,
    'train_sequences': X_train.shape[0],
    'test_sequences': X_test.shape[0],
    'feature_count': len(FEATURE_COLS)
}
log_entry_df = pd.DataFrame([log_entry])

if os.path.exists(log_path):
    log_entry_df.to_csv(log_path, mode='a', header=False, index=False)
else:
    log_entry_df.to_csv(log_path, mode='w', header=True, index=False)



COST_PRICE = 1.50
SALE_PRICE = 1.75
PRICE_MARGIN = SALE_PRICE - COST_PRICE


DAYS_TO_EXPIRE = 3

SIMULATION_START_DATE = test_df.index.min() 
SIMULATION_END_DATE = df.index.max() - pd.Timedelta(days=1) 
LEAD_TIME_DAYS = 1        

simulation_dates = pd.date_range(start=SIMULATION_START_DATE, end=SIMULATION_END_DATE)
results = []

start_stock_val = df.loc[simulation_dates[0], 'stock']
if pd.isna(start_stock_val): start_stock_val = 0

start_stock_val = 0
lstm_stock = max(0, start_stock_val)

for current_date in simulation_dates:
    current_date_str = current_date.strftime('%Y-%m-%d')
    if current_date not in df.index: continue
    next_day = current_date + pd.Timedelta(days=1)
    if next_day not in df.index: break
    
    idx = df.index.get_loc(current_date)
    if idx < SEQ_LEN: continue
    
    actual_sales_next_day = df.loc[next_day, 'Sales_QTTY']

    input_data = df[FEATURE_COLS].iloc[idx-SEQ_LEN+1 : idx+1]
    input_scaled = x_scaler.transform(input_data).reshape(1, SEQ_LEN, len(FEATURE_COLS))
    pred_target = model.predict(input_scaled, verbose=0)[0,0]
    
    if USE_LOG_TARGET and TARGET_TRANSFORM == 'log1p':
        lstm_forecast = max(0, np.expm1(pred_target))
    else:
        lstm_forecast = max(0, pred_target)

    dyn_safety_stock = mae_qty * math.sqrt(LEAD_TIME_DAYS)
    lstm_target = lstm_forecast + dyn_safety_stock
    lstm_order_qty = max(0, lstm_target - lstm_stock)
    
    lstm_total_avail = lstm_stock + lstm_order_qty
    lstm_leftover = max(0, lstm_total_avail - actual_sales_next_day)
    
    lstm_stock = lstm_leftover

    lstm_category = "OPTIMAL"

    if lstm_leftover > 0:
        future_demand_est = df.loc[next_day: min(next_day+pd.Timedelta(days=DAYS_TO_EXPIRE-1), df.index.max()), 'Sales_QTTY'].sum()
        if lstm_leftover > future_demand_est:
            lstm_category = "WASTE"

    lstm_success = 1 if lstm_category == "OPTIMAL" else 0

    results.append({
        'Date': current_date,
        'Stock_Before': round(lstm_total_avail - lstm_order_qty, 1),
        'Forecast': round(lstm_forecast, 1),
        'Safety_Stock': round(dyn_safety_stock, 1),
        'Order_Qty': round(lstm_order_qty, 1),
        'Total_Avail': round(lstm_total_avail, 1),
        'Actual_Sales': actual_sales_next_day,
        'Category': lstm_category,
    })


res_df = pd.DataFrame(results).set_index('Date')

print(res_df.tail())


sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(1, 1, figsize=(16, 7))

ax1.plot(res_df.index, res_df['Actual_Sales'], label='Реални продажби', color='black', linewidth=2, linestyle='-', marker='o', markersize=4, alpha=0.7)
ax1.plot(res_df.index, res_df['Total_Avail'], label='Наличност след поръчка на LSTMwF', color='#2ecc71', linewidth=2, linestyle='--')
ax1.plot(res_df.index, res_df['Forecast'], label='Прогноза на LSTMwF', color='#3498db', linewidth=1, linestyle=':')

# stockout
ax1.fill_between(res_df.index, res_df['Actual_Sales'], res_df['Total_Avail'], 
                 where=(res_df['Actual_Sales'] > res_df['Total_Avail']),
                 color='red', alpha=0.3, label='Дефицит', interpolate=True)

# potentital waste
ax1.fill_between(res_df.index, res_df['Total_Avail'], res_df['Actual_Sales'], 
                 where=(res_df['Total_Avail'] > res_df['Actual_Sales'] * DAYS_TO_EXPIRE),
                 color='orange', alpha=0.2, label='Свръхзапас', interpolate=True)

ax1.set_title(f"Стойности на поръчките и наличностите моделирани от LSTMwF - Продукт {BARCODE}", fontsize=14, fontweight='bold')
ax1.set_ylabel("Брой", fontsize=12)
ax1.legend(loc='upper left', frameon=True)
ax1.grid(True, alpha=0.5)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(f"~/магистър ИИБФ/дипломна/data/plots/backtest_summary_{BARCODE}.svg", format='svg', bbox_inches='tight')
plt.show()


# historical values
history_slice = df.loc[SIMULATION_START_DATE:SIMULATION_END_DATE].copy()
history_slice.info()


sns.set_theme(style="whitegrid")
plt.figure(figsize=(15, 7))

plt.plot(history_slice.index, history_slice['stock'], 
         label='Складова наличност', 
         color='purple', 
         linewidth=2, 
         linestyle='--') 

plt.plot(history_slice.index, history_slice['Sales_QTTY'], 
         label='Реални продажби', 
         color='black', 
         linewidth=1.5, 
         marker='o', 
         markersize=3) 

plt.fill_between(history_slice.index, 0, history_slice['stock'], 
                 where=(history_slice['stock'] <= 0), 
                 color='red', 
                 alpha=0.3, 
                 label='Дефицит', 
                 interpolate=True)

mean_sales = history_slice['Sales_QTTY'].mean()
plt.fill_between(history_slice.index, 
                 history_slice['Sales_QTTY'] * DAYS_TO_EXPIRE, 
                 history_slice['stock'], 
                 where=(history_slice['stock'] > history_slice['Sales_QTTY'] * DAYS_TO_EXPIRE), 
                 color='orange', 
                 alpha=0.1, 
                 label='Свръхзапас', 
                 interpolate=True)


plt.title(f"Складова наличност срещу продажби - Продукт {BARCODE}", 
          fontsize=16, 
          fontweight='bold')
plt.ylabel("Брой", fontsize=12)
plt.xlabel("Дата")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.5)

ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(f"~/магистър ИИБФ/дипломна/data/plots/historical_stock_vs_sales_{BARCODE}.svg", format='svg', bbox_inches='tight')
plt.show()




compare_df = res_df.copy()
compare_df['Historical_Sales'] = df.loc[compare_df.index, 'Sales_QTTY']
compare_df['Historical_Delivery'] = df.loc[compare_df.index, 'Delivery_QTTY']
compare_df['Historical_Revenue'] = compare_df['Historical_Sales'] * SALE_PRICE
compare_df['Historical_COGS'] = compare_df['Historical_Delivery'] * COST_PRICE
compare_df['Historical_Profit'] = compare_df['Historical_Revenue'] - compare_df['Historical_COGS']
s1 = compare_df['Historical_COGS'].sum()

compare_df['Model_Sold'] = compare_df[['Actual_Sales', 'Total_Avail']].min(axis=1)
compare_df['Model_Revenue'] = compare_df['Model_Sold'] * SALE_PRICE
compare_df['Model_COGS'] = compare_df['Order_Qty'] * COST_PRICE
compare_df['Model_Profit'] = compare_df['Model_Revenue'] - compare_df['Model_COGS']
s2 = compare_df['Model_COGS'].sum()

sdiff = s1-s2
print(f"Разлика в разходите за поръчките {sdiff:.2f} лв.")

total_hist_profit = compare_df['Historical_Profit'].sum()
total_model_profit = compare_df['Model_Profit'].sum()
profit_diff = total_model_profit - total_hist_profit

print(f"Историческа печалба:      {total_hist_profit:.2f} лв.")
print(f"Печалба по модела:        {total_model_profit:.2f} лв.")
print(f"Разлика в печалбата между модела и историческите данни: {profit_diff:.2f} лв.")






