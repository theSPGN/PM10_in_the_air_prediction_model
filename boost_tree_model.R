# load used libraries
library(tidymodels) # collection of packages for modeling
library(dplyr) # data manipulation
library(xgboost) # engine for boost_tree model
library(yardstick) # metrics for model
library(vip) # parameters importance

load("prepared_data.RData")
# %%
xgb_mod <-
    boost_tree(
        trees = 1000,
        mtry = 8,
        min_n = tune()
    ) |>
    set_engine(
        engine = "xgboost",
        num.threads = parallel:detectCores() - 1
    ) |>
    set_mode("regression")