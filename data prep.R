library(jsonlite)
library(tidyverse)
library(tidyr)
library(fs)
library(dplyr)
library(readr)
library(zoo)

OPERATION_SALES_TYPE <- "sales"
OPERATION_DELIVERY_TYPE <- "delivery"

process_sales_operations <- function(year_to_process, store = "", op_type, base_path = "~/магистър ИИБФ/дипломна/data/model") {
  operations_folder <- NULL
  if (store != "") {
    operations_folder <- file.path(base_path, as.character(year_to_process), store, "operations")
  } else {
    operations_folder <- file.path(base_path, as.character(year_to_process), "operations")
  }
  
  if (!fs::dir_exists(operations_folder)) {
    stop(paste("Папката за операции за", year_to_process, "не съществува на", operations_folder))
  }

  cat(paste("--- Започва обработка на данни за година", year_to_process, " ", store, "---\n"))
  
  file_list <- c(
    fs::dir_ls(operations_folder, glob = "**/4Operations**", recurse = TRUE)
  ) %>% unique()
  
  if (length(file_list) == 0) {
    cat(paste("Не са намерени JSON файлове в", operations_folder, "\n"))
    return(invisible(NULL))
  }
  
  all_data <- list()
  for (file in file_list) {
   tryCatch({
      data <- jsonlite::fromJSON(file)
      all_data <- append(all_data, list(data))
    }, error = function(e) {
      cat(paste("Грешка при четене на операции от файл", fs::path_file(file), ":", e$message, "\n"))
    })
  }
  
  operations <- dplyr::bind_rows(all_data)
  
  output_path_operations <- file.path(operations_folder, paste0("operations_", year_to_process, "_combined.csv"))
  readr::write_csv(operations, output_path_operations)
  
  cat(paste("Общ брой редове в OPERATIONS:", nrow(operations), "\n"))
  
  sign <- -1
  if (op_type == OPERATION_DELIVERY_TYPE) {
    sign <- 1
  }

  sales <- operations %>%
    dplyr::filter(SIGN == sign) %>%
    dplyr::select(Date, GOODID, QTTY, OPERTYPE, ACCT)
  
  sales <- sales %>%
    dplyr::mutate(
      Date = as.Date(Date, format = "%d.%m.%Y") 
    )
  
  daily_aggregated_sales <- sales %>%
    dplyr::group_by(Date, GOODID) %>%
    dplyr::summarise(
      Total_QTTY = sum(QTTY, na.rm = TRUE),
      Transaction_Count = n(),
      .groups = 'drop'
    ) %>%
    tidyr::complete(
      Date = seq(min(Date), max(Date), by = "day"),
      GOODID,
      fill = list(Total_QTTY = 0, Transaction_Count = 0)
    ) %>%
    dplyr::arrange(Date) %>%
    dplyr::ungroup()
  
  if (op_type == OPERATION_DELIVERY_TYPE) {
    rename_list <- c("Delivery_QTTY" = "Total_QTTY", 
                     "Delivery_Count" = "Transaction_Count")
  } else {
    rename_list <- c("Sales_QTTY" = "Total_QTTY", 
                     "Sales_Count" = "Transaction_Count")
  }

  final_data <- daily_aggregated_sales %>%
    dplyr::rename(!!!rename_list)
  
  output_path_aggregated <- file.path(operations_folder, paste0("daily_aggregated_", op_type, "_full_", year_to_process, ".csv"))
  readr::write_csv(final_data, output_path_aggregated)

  return(invisible(final_data))
}

merge_sales_and_delivery <- function(sales_df, delivery_df) {
  sales_with_delivery <- sales_df %>%
    left_join(delivery_df, by = c("Date", "GOODID"))

  sales_final <- sales_with_delivery %>%
    mutate(
      Delivery_QTTY = replace_na(Delivery_QTTY, 0),
      Delivery_Count = replace_na(Delivery_Count, 0)
    )
  
  return(sales_final)
}

base_dir <- "~/магистър ИИБФ/дипломна/data/model"

# Обработка на 2022 sales
sales_2022_agg <- process_sales_operations(year_to_process = 2022, "", OPERATION_SALES_TYPE, base_path = base_dir)

# Обработка на 2022 delivery
delivery_2022_agg <- process_sales_operations(year_to_process = 2022, "", OPERATION_DELIVERY_TYPE, base_path = base_dir)

sales_delivery_merged_2022 <- merge_sales_and_delivery(
    sales_df = sales_2022_agg,
    delivery_df = delivery_2022_agg)

head(sales_delivery_merged_2022[sales_delivery_merged_2022$Date == '2022-01-02',])

# Обработка на 2023 sales
sales_2023_agg <- process_sales_operations(year_to_process = 2023, "", OPERATION_SALES_TYPE, base_path = base_dir)
head(sales_2023_agg[sales_2023_agg$Date == '2023-01-03',] )
# Обработка на 2023 delivery
delivery_2023_agg <- process_sales_operations(year_to_process = 2023, "", OPERATION_DELIVERY_TYPE, base_path = base_dir)
head(delivery_2023_agg[delivery_2023_agg$Delivery_QTTY != 0,])

sales_delivery_merged_2023 <- merge_sales_and_delivery(
  sales_df = sales_2023_agg,
  delivery_df = delivery_2023_agg)

head(sales_delivery_merged_2023[sales_delivery_merged_2023$Date == '2023-01-04' & sales_delivery_merged_2023$Delivery_QTTY != 0,])


# Обработка на 2024 sales
sales_2024_agg <- process_sales_operations(year_to_process = 2024, "", OPERATION_SALES_TYPE, base_path = base_dir)
head(sales_2024_agg)
# Обработка на 2024 delivery
delivery_2024_agg <- process_sales_operations(year_to_process = 2024, "", OPERATION_DELIVERY_TYPE, base_path = base_dir)
head(delivery_2024_agg[delivery_2024_agg$Delivery_QTTY != 0,])

sales_delivery_merged_2024 <- merge_sales_and_delivery(
  sales_df = sales_2024_agg,
  delivery_df = delivery_2024_agg)

head(sales_delivery_merged_2024[sales_delivery_merged_2024$Date == '2024-01-04' & sales_delivery_merged_2024$Delivery_QTTY != 0,])
tail(sales_delivery_merged_2024[sales_delivery_merged_2024$Total_QTTY != 0,])
tail(delivery_2024_agg)

# fill in the missing data 10.10.2024 - 31.12.2024
tail(sales_2023_agg)

avg_pattern_2023 <- sales_2023_agg %>%
  mutate(month = month(Date), day = day(Date)) %>%
  group_by(GOODID, month, day) %>%
  summarise(
    Sales_QTTY = mean(Sales_QTTY, na.rm = TRUE),
    Sales_Count = mean(Sales_Count, na.rm = TRUE),
    .groups = "drop"
  )

missing_dates <- seq(as.Date("2024-10-11"), as.Date("2024-12-31"), by = "day")
all_goods <- unique(sales_2024_agg$GOODID)

missing_grid <- expand.grid(Date = missing_dates, GOODID = all_goods) %>%
  mutate(month = month(Date), day = day(Date))


filled_missing <- missing_grid %>%
  left_join(avg_pattern_2023, by = c("GOODID", "month", "day")) %>%
  mutate(
    Sales_QTTY = ifelse(is.na(Sales_QTTY), 0, Sales_QTTY),
    Sales_Count = ifelse(is.na(Sales_Count), 0, Sales_Count)
  ) %>%
  select(Date, GOODID, Sales_QTTY, Sales_Count)

sales_2024_agg_filled <- bind_rows(sales_2024_agg, filled_missing) %>%
  arrange(Date)
range(sales_2024_agg_filled$Date)

#---
avg_delivery_pattern_2023 <- delivery_2023_agg %>%
  mutate(month = month(Date), day = day(Date)) %>%
  group_by(GOODID, month, day) %>%
  summarise(
    Delivery_QTTY = mean(Delivery_QTTY, na.rm = TRUE),
    Delivery_Count = mean(Delivery_Count, na.rm = TRUE),
    .groups = "drop"
  )

missing_dates <- seq(as.Date("2024-10-11"), as.Date("2024-12-31"), by = "day")
all_goods_delivery <- unique(delivery_2024_agg$GOODID)

missing_grid_delivery <- expand.grid(Date = missing_dates, GOODID = all_goods_delivery) %>%
  mutate(month = month(Date), day = day(Date))


filled_missing_delivery <- missing_grid_delivery %>%
  left_join(avg_delivery_pattern_2023, by = c("GOODID", "month", "day")) %>%
  mutate(
    Delivery_QTTY = ifelse(is.na(Delivery_QTTY), 0, Delivery_QTTY),
    Delivery_Count = ifelse(is.na(Delivery_Count), 0, Delivery_Count)
  ) %>%
  select(Date, GOODID, Delivery_QTTY, Delivery_Count)

delivery_2024_agg_filled <- bind_rows(delivery_2024_agg, filled_missing_delivery) %>%
  arrange(Date)
range(delivery_2024_agg_filled$Date)

sales_delivery_merged_2024_filled <- merge_sales_and_delivery(
  sales_df = sales_2024_agg_filled,
  delivery_df = delivery_2024_agg_filled)

head(sales_delivery_merged_2024_filled[sales_delivery_merged_2024_filled$Date == '2024-01-04' & sales_delivery_merged_2024_filled$Delivery_QTTY != 0,])
tail(sales_delivery_merged_2024_filled[sales_delivery_merged_2024_filled$Delivery_QTTY != 0,])
tail(delivery_2024_agg_filled[delivery_2024_agg_filled$Delivery_QTTY != 0, ])

#readr::write_csv(sales_2024_agg_filled, final_output_path)

all_sales_combined <- dplyr::bind_rows(sales_2022_agg, sales_2023_agg, sales_2024_agg_filled) %>%
 dplyr::arrange(Date)

cat(paste("\nTotal rows in combined dataset (2022 & 2023 & 2024):", nrow(all_sales_combined), "\n"))

output_folder <- file.path(base_dir, "processed")
final_output_path <- file.path(output_folder, "daily_sales_2022_2023_2024_combined_full.csv")

readr::write_csv(all_sales_combined, final_output_path)

all_sales_and_deliveries_combined <- dplyr::bind_rows(sales_delivery_merged_2022, sales_delivery_merged_2023, sales_delivery_merged_2024_filled) %>%
  dplyr::arrange(Date)

cat(paste("\nTotal rows in combined dataset (2022 & 2023 & 2024):", nrow(all_sales_and_deliveries_combined), "\n"))

output_folder <- file.path(base_dir, "processed")
final_output_path_sales_deliveries <- file.path(output_folder, "daily_sales_and_deliveries_2022_2023_2024_combined_full.csv")

readr::write_csv(all_sales_and_deliveries_combined, final_output_path_sales_deliveries)
range(all_sales_and_deliveries_combined$Date)
head(all_sales_and_deliveries_combined[all_sales_and_deliveries_combined$Date == '2022-01-02',])

# check
example_good <- sample(unique(sales_2024_agg$GOODID), 2)

sales_2024_plot <- sales_2024_agg %>%
  filter(GOODID == example_good) %>%
  mutate(Source = "Original")

sales_2024_filled_plot <- sales_2024_agg_filled %>%
  filter(GOODID == example_good) %>%
  mutate(Source = "Filled")


plot_data <- bind_rows(sales_2024_plot, sales_2024_filled_plot)

ggplot(plot_data, aes(x = Date, y = Total_QTTY, color = Source)) +
  geom_line(size = 1) +
  labs(
    title = paste("Продажби за продукт", example_good),
    subtitle = "Оригинални vs Попълнени с данни от 2023",
    x = "Дата",
    y = "Количество (Total_QTTY)"
  ) +
  theme_minimal() +
  scale_color_manual(values = c("Original" = "#FF5733", "Filled" = "#2E86C1"))


# Обработка на 2025 mag1
sales_2025_agg_mag1 <- process_sales_operations(year_to_process = 2025, store = "mag1", OPERATION_SALES_TYPE,  base_path = base_dir)

# Обработка на 2025 mag2
sales_2025_agg_mag2 <- process_sales_operations(year_to_process = 2025, store = "mag2", OPERATION_SALES_TYPE, base_path = base_dir)

head(sales_2025_agg_mag1)
head(sales_2025_agg_mag2)

process_goods_data <- function(year_to_process,
                               store = "",
                               base_path = "~/магистър ИИБФ/дипломна/data/model") {

  goods_folder <- if (store != "") {
    file.path(base_path, as.character(year_to_process), store, "goods")
  } else {
    file.path(base_path, as.character(year_to_process), "goods")
  }
  
  if (!fs::dir_exists(goods_folder)) {
    stop(paste("Папката за goods за", year_to_process, "не съществува на", goods_folder))
  }
  
  cat(paste("--- Започва обработка на GOODS данни за година", year_to_process, store, "---\n"))
  
  goods_file_list <- fs::dir_ls(goods_folder, glob = "**/3Goods**", recurse = TRUE) %>% unique()
  
  if (length(goods_file_list) == 0) {
    cat(paste("Не са намерени JSON файлове в", goods_folder, "\n"))
    return(invisible(NULL))
  }
  
  goods_list <- list()
  for (file in goods_file_list) {
    tryCatch({
      df <- jsonlite::fromJSON(file) %>% as_tibble()
      required_cols <- c("ID", "CODE", "BARCODE1", "NAME", "MEASURE1", "PRICEOUT1")
      existing_cols <- intersect(required_cols, names(df))
      df <- df %>% select(all_of(existing_cols))
      
      goods_list[[file]] <- df
    }, error = function(e) {
      cat(paste("Пропуснат файл", basename(file), ":", e$message, "\n"))
    })
  }
  
  combined_goods <- dplyr::bind_rows(goods_list)
  
  cat(paste("Общ брой редове в обединените данни:", nrow(combined_goods), "\n"))

  combined_unique_goods <- combined_goods %>%
    distinct(ID, .keep_all = TRUE)
  
  cat(paste("Уникални продукти:", nrow(combined_unique_goods), "\n"))
  
  output_file <- if (store != "") {
    file.path(goods_folder, paste0("goods_", store, "_", year_to_process, ".csv"))
  } else {
    file.path(goods_folder, paste0("goods_", year_to_process, ".csv"))
  }
  
  readr::write_csv(combined_unique_goods, output_file)
  
  return(invisible(combined_unique_goods))
}

base_dir <- "~/магистър ИИБФ/дипломна/data/model"

goods_2024 <- process_goods_data(2024, base_path = base_dir)
goods_2025_mag1 <- process_goods_data(2025, store = "mag1", base_path = base_dir)
goods_2025_mag2 <- process_goods_data(2025, store = "mag2", base_path = base_dir)

unify_good_ids_across_stores <- function(goods_2024_path,
                                         goods_mag1_path,
                                         goods_mag2_path,
                                         sales_mag1_path,
                                         sales_mag2_path,
                                         output_path) {
  
  goods_2024 <- read_csv(goods_2024_path, show_col_types = FALSE) %>%
    select(ID_2024 = ID, BARCODE1)
  
  goods_mag1 <- read_csv(goods_mag1_path, show_col_types = FALSE) %>%
    select(ID_mag1 = ID, BARCODE1)
  
  goods_mag2 <- read_csv(goods_mag2_path, show_col_types = FALSE) %>%
    select(ID_mag2 = ID, BARCODE1)

  sales_mag1 <- read_csv(sales_mag1_path, show_col_types = FALSE)
  sales_mag2 <- read_csv(sales_mag2_path, show_col_types = FALSE)
  
  cat(paste("  Маг1 продажби:", nrow(sales_mag1), "реда\n"))
  cat(paste("  Маг2 продажби:", nrow(sales_mag2), "реда\n"))

  map_mag1 <- goods_mag1 %>%
    left_join(goods_2024, by = "BARCODE1") %>%
    filter(!is.na(ID_2024))
  
  map_mag2 <- goods_mag2 %>%
    left_join(goods_2024, by = "BARCODE1") %>%
    filter(!is.na(ID_2024))

  sales_mag1_mapped <- sales_mag1 %>%
    left_join(map_mag1, by = c("GOODID" = "ID_mag1")) %>%
    mutate(GOODID = coalesce(ID_2024, GOODID)) %>%
    select(-ID_2024, -BARCODE1)
  
  sales_mag2_mapped <- sales_mag2 %>%
    left_join(map_mag2, by = c("GOODID" = "ID_mag2")) %>%
    mutate(GOODID = coalesce(ID_2024, GOODID)) %>%
    select(-ID_2024, -BARCODE1)

  sales_2025_combined <- bind_rows(
    sales_mag1_mapped %>% mutate(Store = "mag1"),
    sales_mag2_mapped %>% mutate(Store = "mag2")
  ) %>%
    arrange(Date, GOODID)
  
  cat(paste("  Комбиниран файл:", nrow(sales_2025_combined), "реда\n"))

  write_csv(sales_2025_combined, output_path)

  return(sales_2025_combined)
}

base_dir <- "~/магистър ИИБФ/дипломна/data/model"

goods_2024_path <- file.path(base_dir, "2024/goods/goods_2024.csv")
goods_mag1_path <- file.path(base_dir, "2025/mag1/goods/goods_mag1_2025.csv")
goods_mag2_path <- file.path(base_dir, "2025/mag2/goods/goods_mag2_2025.csv")

sales_mag1_path <- file.path(base_dir, "2025/mag1/operations/daily_aggregated_sales_full_2025.csv")
sales_mag2_path <- file.path(base_dir, "2025/mag2/operations/daily_aggregated_sales_full_2025.csv")

output_path <- file.path(base_dir, "2025/sales_2025_combined.csv")

sales_2025_combined <- unify_good_ids_across_stores(
  goods_2024_path = goods_2024_path,
  goods_mag1_path = goods_mag1_path,
  goods_mag2_path = goods_mag2_path,
  sales_mag1_path = sales_mag1_path,
  sales_mag2_path = sales_mag2_path,
  output_path = output_path
)

sales_2025_combined <- sales_2025_combined %>%
  mutate(Date = as.Date(Date, format = "%Y-%m-%d"))

sales_2025_daily <- sales_2025_combined %>%
  group_by(Date, GOODID) %>%
  summarise(
    Sales_QTTY = sum(Sales_QTTY, na.rm = TRUE),
    Sales_Count = sum(Sales_Count, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  complete(
    Date = seq(min(Date), max(Date), by = "day"),
    GOODID,
    fill = list(Sales_QTTY = 0, Sales_Count = 0)
  ) %>%
  arrange(Date, GOODID)

head(sales_2025_daily)
range(sales_2025_daily$Date)

# Обработка на 2025 delivery
delivery_2025_agg_mag1 <- process_sales_operations(year_to_process = 2025, "mag1", OPERATION_DELIVERY_TYPE, base_path = base_dir)
head(delivery_2025_agg_mag1[delivery_2025_agg_mag1$Delivery_QTTY != 0,])
delivery_2025_agg_mag2 <- process_sales_operations(year_to_process = 2025, "mag2", OPERATION_DELIVERY_TYPE, base_path = base_dir)
head(delivery_2025_agg_mag2[delivery_2025_agg_mag2$Delivery_QTTY != 0,])


delivery_mag1_path <- file.path(base_dir, "2025/mag1/operations/daily_aggregated_delivery_full_2025.csv")
delivery_mag2_path <- file.path(base_dir, "2025/mag2/operations/daily_aggregated_delivery_full_2025.csv")

output_path <- file.path(base_dir, "2025/delivery_2025_combined.csv")

delivery_2025_combined <- unify_good_ids_across_stores(
  goods_2024_path = goods_2024_path,
  goods_mag1_path = goods_mag1_path,
  goods_mag2_path = goods_mag2_path,
  sales_mag1_path = delivery_mag1_path,
  sales_mag2_path = delivery_mag2_path,
  output_path = output_path
)

delivery_2025_daily <- delivery_2025_combined %>%
  group_by(Date, GOODID) %>%
  summarise(
    Delivery_QTTY = sum(Delivery_QTTY, na.rm = TRUE),
    Delivery_Count = sum(Delivery_Count, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  complete(
    Date = seq(min(Date), max(Date), by = "day"),
    GOODID,
    fill = list(Delivery_QTTY = 0, Delivery_Count = 0)
  ) %>%
  arrange(Date, GOODID)


sales_delivery_merged_2025 <- merge_sales_and_delivery(
  sales_df = sales_2025_daily,
  delivery_df = delivery_2025_daily)

sales_delivery_merged_2025[sales_delivery_merged_2025$Delivery_QTTY != 0,]

all_sales_combined_final <- dplyr::bind_rows(sales_2022_agg, sales_2023_agg, sales_2024_agg_filled, sales_2025_daily) %>%
  dplyr::arrange(Date)

cat(paste("\nTotal rows in combined dataset (2022 & 2023 & 2024 & 2025):", nrow(all_sales_combined_final), "\n"))

output_folder <- file.path(base_dir, "processed")
final_output_path <- file.path(output_folder, "daily_sales_2022_2023_2024_2025_combined_full.csv")

readr::write_csv(all_sales_combined_final, final_output_path)


all_sales_and_delivery_combined_final <- dplyr::bind_rows(sales_delivery_merged_2022, sales_delivery_merged_2023, sales_delivery_merged_2024_filled, sales_delivery_merged_2025) %>%
  dplyr::arrange(Date)

cat(paste("\nTotal rows in combined dataset (sales & delivery) (2022 & 2023 & 2024 & 2025):", nrow(all_sales_and_delivery_combined_final), "\n"))

output_folder <- file.path(base_dir, "processed")
final_output_path_sales_deliveries_all <- file.path(output_folder, "daily_sales_and_deliveries_2022_2023_2024_2025_combined_full.csv")

readr::write_csv(all_sales_and_delivery_combined_final, final_output_path_sales_deliveries_all)





