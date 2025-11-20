#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 13:02:18 2025

@author: I568766
"""
import pandas as pd
import os
import json
import glob
import numpy as np

# folder with goods raw data
goods_raw_folder = '/Users/I568766/магистър ИИБФ/дипломна/data/model/goods_raw'

# all files "3Goods" (json or txt)
goods_raw_folder_file_list = glob.glob(os.path.join(goods_raw_folder, '**/3Goods.json'), recursive=True) + \
             glob.glob(os.path.join(goods_raw_folder, '**/3Goods**'), recursive=True)

print(f"Намерени файлове: {len(goods_raw_folder_file_list)}")
for f in goods_raw_folder_file_list:
    print(" -", f)


dfs = []
for path in goods_raw_folder_file_list:
    try:
        df = pd.read_json(path, encoding='utf-8')
        # филтрираме само нужните колони
        df = df[['ID','CODE','BARCODE1', 'NAME', 'MEASURE1','MINQTTY']]
        #df['source_file'] = os.path.basename(path)   # добавяме име на източник
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ Пропуснат файл {path}: {e}")

# 🧮 Комбинираме всички DataFrame-и
combined = pd.concat(dfs, ignore_index=True)

# ❌ Премахваме дубликати само ако има напълно еднакви двойки (ID, BARCODE1)
combined = combined.drop_duplicates(subset=['ID', 'BARCODE1'], keep='first')

# 💾 Запис в CSV
out_path = '/Users/I568766/магистър ИИБФ/дипломна/data/model/processed/goods.csv'
combined.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"\n✅ Записано: {out_path}")
print(f"Общо уникални двойки ID–BARCODE1: {len(combined)}")


operations_raw_folder = '/Users/I568766/магистър ИИБФ/дипломна/data/model/operations_raw'

# Collect all .txt file paths
operations_raw_folder_file_list = glob.glob(os.path.join(operations_raw_folder, '**/3Operations.json'), recursive=True) + \
             glob.glob(os.path.join(operations_raw_folder, '**/4Operations**'), recursive=True)
all_data = []

# Read and append JSON arrays from each file
for file in operations_raw_folder_file_list:
    print(file)
    with open(file, "r") as f:
        data = json.load(f)  # Parse the JSON array
        all_data.extend(data)  # Append all items

# Convert to DataFrame
operations = pd.DataFrame(all_data)

# Save to CSV
output_path = "/Users/I568766/магистър ИИБФ/дипломна/data/model/operations_raw/operations_all_mag2.csv"
operations.to_csv(output_path, index=False)

sales = operations[operations['SIGN'] == -1]
sales.to_csv('/Users/I568766/магистър ИИБФ/дипломна/data/model/operations_raw/sales_raw.csv', index=False)
len(sales)

len(combined)
combined_unique = combined.drop_duplicates(subset='ID', keep='first')
len(combined_unique)
merged = sales.merge(combined_unique[['ID', 'BARCODE1', 'NAME']], left_on='GOODID', right_on='ID', how='left').drop(columns=['ID'])

merged.columns

merged = merged[['Date','BARCODE1', 'NAME','QTTY','PRICEOUT', 'OPERTYPE', 'ACCT', 'GOODID']]
merged['Date'] = pd.to_datetime(merged['Date'], dayfirst=True, errors='coerce')
merged = merged.sort_values(by='Date', ascending=True).reset_index(drop=True)

merged.to_csv('/Users/I568766/магистър ИИБФ/дипломна/data/model/operations_raw/sales_mag1.csv', index=False)

# Path to the second dataset
sales_mag1_path = "/Users/I568766/магистър ИИБФ/дипломна/data/model/operations_raw/sales-01-10.csv"

# Read the second dataset
sales_mag1 = pd.read_csv(sales_mag1_path)
sales_mag1['Date'] = pd.to_datetime(sales_mag1['Date'], dayfirst=True, errors='coerce')
merged1 = sales_mag1.merge(combined_unique[['ID', 'BARCODE1', 'NAME']], left_on='GOODID', right_on='ID', how='left').drop(columns=['ID'])



combined_sales = pd.concat([merged, merged1], ignore_index=True)
len(combined_sales)

len(merged) + len(merged1) == len(combined_sales)
combined_sales = combined_sales.sort_values(by='Date', ascending=True).reset_index(drop=True)

final_sales_path = "/Users/I568766/магистър ИИБФ/дипломна/data/model/processed/sales_2025.csv"
combined_sales.to_csv(final_sales_path, index=False)


# goods from mag1 + goods from mag2 -> combined goods [ID, Barcode]
# merge with sales_2025 to fill the gaps of products
# read the xlsx files to calculate the lot

# model??
