import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


df = pd.read_csv('~/магистър ИИБФ/дипломна/data/model/processed/daily_sales_and_deliveries_2022_2023_2024_2025_combined_full.csv',parse_dates=True)

df.info()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

print("--- Анализ на Липсващи Стойности ---")
missing_values = df.isnull().sum()
print(missing_values)

print("\n--- Описателни Статистики ---")
print(df[df['Sales_QTTY'] > 0][['Sales_QTTY', 'Sales_Count']].describe())
print(df[df['Delivery_QTTY'] > 0][['Delivery_QTTY', 'Delivery_Count']].describe())


daily_agg = df.resample('D').agg({
    'Sales_QTTY': 'sum',
    'Delivery_QTTY': 'sum',
    'Sales_Count': 'sum',
    'Delivery_Count': 'sum'
})

plt.figure(figsize=(16, 6))
daily_agg['Sales_QTTY'].plot(title='Общи дневни продажби във времето')
plt.ylabel('Общо количество продажби')
plt.show()


low_sales_dates_df = daily_agg[daily_agg['Sales_QTTY'] <= 8]
low_sales_dates = low_sales_dates_df.index
print("\n--- Дати без продажби или минимални продажби ---")
print(low_sales_dates.strftime('%Y-%m-%d').tolist())


date_of_max_sales = daily_agg['Sales_QTTY'].idxmax()
max_sales_quantity = daily_agg['Sales_QTTY'].max()
print("--- Дата с максимални дневни продажби ---")
print(f"Дата: {date_of_max_sales.strftime('%Y-%m-%d')}")
print(f"Максимално количество (Sales_QTTY): {max_sales_quantity:.2f} бр.")

# seasonal decomposition
decomposition_yearly = seasonal_decompose(daily_agg['Sales_QTTY'], model='additive', period=365)
fig = decomposition_yearly.plot()
fig.axes[0].set_title('Продажби', fontsize=12)
fig.axes[1].set_title('Тренд', fontsize=12)
fig.axes[2].set_title('Сезонност', fontsize=12)
fig.axes[3].set_title('Остатъци', fontsize=12)

fig.set_size_inches(14, 10)
plt.suptitle('Годишна сезонна декомпозиция', y=1.02, fontsize=16)
fig.savefig('/Users/I568766/магистър ИИБФ/дипломна/data/plots/seasonal_decomposition_yearly.svg', format='svg')
plt.show()

# autocorrelation of sales
max_lag = 40
N_TOP_LAGS = 5


acf_values, conf_int = acf(daily_agg['Sales_QTTY'].dropna(), nlags=max_lag, alpha=0.05)

acf_df = pd.DataFrame({
    'Lag': range(max_lag + 1),
    'ACF': acf_values,
    'Lower_CI': conf_int[:, 0] - acf_values,
    'Upper_CI': conf_int[:, 1] - acf_values
})

acf_df = acf_df.iloc[1:]

significant_lags = acf_df[
    (acf_df['ACF'] > acf_df['Upper_CI']) | (acf_df['ACF'] < acf_df['Lower_CI'])
].copy()

significant_lags['Abs_ACF'] = significant_lags['ACF'].abs()
top_lags = significant_lags.nlargest(N_TOP_LAGS, 'Abs_ACF')

print(top_lags[['Lag', 'ACF']])

fig, ax = plt.subplots(figsize=(14, 7))
plot_acf(daily_agg['Sales_QTTY'].dropna(), lags=max_lag, ax=ax, title='Автокорелация на дневни продажби')

for index, row in top_lags.iterrows():
    lag = int(row['Lag'])
    acf_value = row['ACF']
    
    ax.axvline(lag, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    
    text_y_position = -0.15
    
    ax.text(lag, text_y_position,
            f'Lag {lag}\n({acf_value:.2f})', 
            color='red', 
            fontsize=10,
            ha='center',
            va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
    
ticks = sorted(list(set([0, 1] + top_lags['Lag'].tolist() + [max_lag])))
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
ax.set_xlabel('Лаг (Дни)')
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
fig.savefig('/Users/I568766/магистър ИИБФ/дипломна/data/plots/autocorrelation_sales.svg', format='svg')
plt.show()


