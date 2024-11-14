library(tidymodels)
library(dplyr)
library(ranger)
library(yardstick)
library(vip)

# Seed
set.seed(213)

# Load data
load("prepared_data.RData")

# Random forest model
rf_model <- rand_forest(
    mode = "regression",
    mtry = tune(),
    min_n = tune(),
    trees = tune()
) |>
    set_engine("ranger",
        num.threads = parallel::detectCores() - 1
    ) |>
    set_mode("regression")

id_variables <- train_data |>
    select_if(is.numeric()) |>
    cor(use = "complete.obs") |>
    as.data.frame() |>
    select(grimm_pm10) |>
    abs() |>
    filter(grimm_pm10 < 0.3) |>
    rownames()
