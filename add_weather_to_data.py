import pandas as pd


data_path = "~/магистър ИИБФ/дипломна/data/model/processed/daily_sales_2000008.csv"
weather_path = "~/магистър ИИБФ/дипломна/data/model/processed/weather_data.csv"
output_path = "~/магистър ИИБФ/дипломна/data/model/processed/daily_sales_2000008_with_weather.csv"


df = pd.read_csv(
    filepath_or_buffer=data_path,
    usecols=["Date", "Sales_QTTY", "Delivery_QTTY"],
    parse_dates=["Date"],
    date_format="%Y-%m-%d"
)

print(df.head())

weather_data = pd.read_csv(
    filepath_or_buffer=weather_path,
    parse_dates=["Date"],
    date_format="%Y-%m-%d"
)

print(weather_data.head())


final_df = pd.merge(
    left=df,
    right=weather_data,
    on="Date",
    how="left"
)

print(final_df.head())

final_df.to_csv(
    path_or_buf=output_path,
    index=False
)
