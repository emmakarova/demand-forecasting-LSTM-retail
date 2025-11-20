library(httr)
library(jsonlite)
library(dplyr)
library(lubridate)
library(readr)

fetch_weather_data_r <- function(start_date, end_date, latitude = 44.020833, longitude = 27.001389) {
  base_url <- "https://archive-api.open-meteo.com/v1/archive"
  
  params <- list(
    latitude = latitude,
    longitude = longitude,
    start_date = start_date,
    end_date = end_date,
    daily = paste("temperature_2m_max", "precipitation_sum", "sunshine_duration", sep = ","),
    timezone = "Europe/Sofia"
  )
  
  response <- GET(url = base_url, query = params)

  if (http_error(response)) {
    stop(paste("HTTP грешка:", status_code(response)))
  }
  
  data <- fromJSON(content(response, "text", encoding = "UTF-8"))
  
  daily_data <- data$daily
  
  weather_df <- as.data.frame(daily_data)
  
  weather_df <- weather_df %>%
    rename(
      Date = time,
      max_temp = temperature_2m_max,
      precipitation = precipitation_sum,
      sunshine_duration_s = sunshine_duration
    ) %>%
    mutate(
      Date = as.Date(Date),
      sunshine_hours = sunshine_duration_s / 3600
    ) %>%
    select(Date, max_temp, precipitation, sunshine_hours)
  
  return(weather_df)
}

start_date_r <- "2022-01-01"
end_date_r <- "2025-10-22"

weather_data_r <- fetch_weather_data_r(start_date_r, end_date_r)

write_csv(weather_data_r, "~/магистър ИИБФ/дипломна/data/model/processed/weather_data.csv")



