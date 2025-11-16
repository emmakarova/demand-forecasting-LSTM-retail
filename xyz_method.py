import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE_PATH = '~/магистър ИИБФ/дипломна/data/model/processed/daily_sales_and_deliveries_2022_2023_2024_2025_combined_full.csv'
OUTPUT_DIR = '~/магистър ИИБФ/дипломна/data/model/processed/segmented_xyz/'

THRESHOLD_Y = 0.40
THRESHOLD_Z = 0.70


os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE_PATH, parse_dates=['Date'])

if df['Sales_QTTY'].isnull().any():
    df['Sales_QTTY'].fillna(0, inplace=True)


product_metrics = df.groupby('GOODID')['Sales_QTTY'].agg(
    mean_qty='mean',
    std_dev_qty='std',
    sum_qty='sum',
    count_days='count'
).reset_index()


product_metrics['coef_var'] = np.where(
    product_metrics['mean_qty'] > 0,
    product_metrics['std_dev_qty'] / product_metrics['mean_qty'],
    np.inf 
)

def classify_xyz(coef_var):
    if coef_var < THRESHOLD_Y:
        return 'X'
    elif coef_var < THRESHOLD_Z:
        return 'Y'
    else:
        return 'Z'

product_metrics['XYZ_category'] = product_metrics['coef_var'].apply(classify_xyz)

TOTAL_PRODUCTS = len(product_metrics)
TOTAL_SALES_QTTY = product_metrics['sum_qty'].sum()

xyz_summary = product_metrics.groupby('XYZ_category').agg(
    product_count=('GOODID', 'count'),
    median_coef_var=('coef_var', 'median')
).reset_index()

xyz_summary['percentage_of_all_products'] = (xyz_summary['product_count'] / TOTAL_PRODUCTS) * 100

print(xyz_summary)
print("\n--- Резултати от XYZ Анализа ---")
print(product_metrics[['GOODID', 'mean_qty', 'coef_var', 'XYZ_category']].head())
print("\nРазпределение по категории:")
#print(product_metrics['XYZ_category'].value_counts())
category_counts = product_metrics['XYZ_category'].value_counts()
print(category_counts)

df = df.merge(
    product_metrics[['GOODID', 'XYZ_category']], 
    on='GOODID', 
    how='left'
)


max_cv_display = 3.0 
cv_filtered = product_metrics[
    (product_metrics['coef_var'] < max_cv_display) & 
    (product_metrics['coef_var'] != np.inf)
]['coef_var']

# 2. Построяване на Хистограмата
plt.figure(figsize=(12, 6))
sns.histplot(cv_filtered, 
             bins=50, 
             kde=True, 
             log_scale=(False, True), # Log scale на Y-ос за по-добро виждане на редките стойности
             color='skyblue',
             edgecolor='black')

# 3. Добавяне на праговите линии (Cut-off Points)
# Линия за X/Y праг (THRESHOLD_Y = 0.50)
plt.axvline(x=THRESHOLD_Y, 
            color='green', 
            linestyle='--', 
            linewidth=2, 
            label=f'X/Y праг ({THRESHOLD_Y:.2f})')
plt.text(THRESHOLD_Y + 0.05, plt.ylim()[1]*0.8, 
         f'X: {category_counts.get("X", 0)} бр.', color='green', fontsize=12, ha='left')

# Линия за Y/Z праг (THRESHOLD_Z = 1.0)
plt.axvline(x=THRESHOLD_Z, 
            color='orange', 
            linestyle='--', 
            linewidth=2, 
            label=f'Y/Z праг ({THRESHOLD_Z:.2f})')
plt.text(THRESHOLD_Z + 0.05, plt.ylim()[1]*0.7, 
         f'Y: {category_counts.get("Y", 0)} бр.', color='orange', fontsize=12, ha='left')


# 4. Форматиране
plt.title('Разпределение на Коефициента на Вариация (V) на Търсенето', fontsize=16)
plt.xlabel('Коефициент на Вариация (V) – $\sigma / \mu$', fontsize=14)
plt.ylabel('Брой Артикули (Log Scale)', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim(0, max_cv_display) # Ограничаваме x-оста до 3.0
plt.tight_layout()

# 5. Запазване на графиката
histogram_output_path = os.path.join('/Users/I568766/магистър ИИБФ/дипломна/data/plots/', f'xyz_v_distribution_histogram_{THRESHOLD_Y}/{THRESHOLD_Z}.svg')
plt.savefig(f'/Users/I568766/магистър ИИБФ/дипломна/data/plots/xyz_v_distribution_histogram_{THRESHOLD_Y}:{THRESHOLD_Z}.svg')
print(f"\nХистограмата е запазена като: {histogram_output_path}")

plt.show()


for category in ['X', 'Y', 'Z']:
   product_df_category = product_metrics[product_metrics['XYZ_category'] == category].drop(columns=['XYZ_category'])
    
   output_path_metrics = os.path.join(OUTPUT_DIR, f'product_metrics_xyz_full__{category}.csv')

   product_df_category.to_csv(output_path_metrics, index=False)
    
   print(f"Категория {category} (МЕТРИКИ): {len(product_df_category)} уникални продукта, запазени във {output_path_metrics}")

