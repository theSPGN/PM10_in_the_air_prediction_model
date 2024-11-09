library(tidyverse)
library(openair)
library(ggplot2)
library(GGally)

load("ops.RData")
ops <- ops |> na.omit()

load("data_test.rdata")

ops |> select(-date) |> ggpairs()

ops |> select_if(is.numeric) |> 
  cor(use = "complete.obs") |> as.data.frame() |> select(grimm_pm10)

ops |> select_if(is.numeric) |> 
  cor(use = "complete.obs") |> as.data.frame() |> select(ops_pm10)

ops <- ops |> select(-pres_sea)

# step_pca, step_corr, step_date(month), step_time(hour) to recipe
# update_role(grimm_pm10, rh, temp, pres, wd, prec, new_role="ID")

val_set_grimm <- validation_split(
  data = ops,
  prop = 3 / 4,
  strata = grimm_pm10
)

val_set_pm10 <- validation_split(
  data = ops,
  prop = 3 / 4,
  strata = ops_pm10
)

test_data <- 
  ops_data |> 
  mutate(
    ops_pm10 = ops_bam$ops_pm10[1:nrow(ops_data)]) |> 
  select(colnames(ops))

# resample = val_set
# tune - 1 parametr modelu
# set_engine(num.threads = parallel::detectCores() - 1,) 

# Jakub - decision_tree()
# Daria - random_forest()
# Maria - cubist_rules()
# Mateusz - xgboost()