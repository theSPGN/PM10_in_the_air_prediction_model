library(tidyverse)
library(tidymodels)
library(openair)
library(ggplot2)
library(GGally)

# Wczytanie danych do trenowania modelu
load("ops.RData")
ops <- ops |> na.omit()

# Wczytanie danych z zewnętrznej stacji
# nie jest to zbiór testowy
load("data_test.rdata")

# Wykresy korelacji danych
# ops |> select(-date) |> ggpairs()

# Korelacje ze zmiennymi dla grimm_pm10
ops |>
  select_if(is.numeric) |>
  cor(use = "complete.obs") |>
  as.data.frame() |>
  select(grimm_pm10)

# Usuwamy pres_sea ponieważ nie ma w danych z zewnętrznej stacji
# pres_sea jest skorelowane mocno z pres
ops |>
  select_if(is.numeric) |>
  cor(use = "complete.obs") |>
  as.data.frame() |>
  rownames_to_column(var = "rowname") |>
  filter(rowname == "pres") |>
  select(pres_sea)
model_station_data <- ops |> select(-pres_sea)

# wyjście grimm_pm10
# step_pca/step_corr, step_date(month), step_time(hour), step_rm(date) użyć tych kroków na pewno proponuję
# update_role(ops_pm10, rh, temp, wd, prec, new_role="ID") bo są słabo skorelowane lub za bardzo

data_split <- initial_split(
  data = model_station_data,
  prop = 3 / 4,
  strata = grimm_pm10
)
train_data <- training(data_split)
test_data <- testing(data_split)

val_set <- validation_split(
  data = train_data,
  prop = 3 / 4,
  strata = grimm_pm10
)


# Nie używamy póki co (jak skończymy modele to wtedy coś z tym będziemy robić)
other_station_data <-
  ops_data |>
  mutate(
    ops_pm10 = ops_bam$ops_pm10[1:nrow(ops_data)]
  ) |>
  select(colnames(model_station_data))


rm(list = c("ops", "bam", "ops_bam", "ops_data", "data_split", "model_station_data"))

save(train_data, test_data, val_set, other_station_data, file = "prepared_data.RData")

# resample = val_set
# tune - 1 parametr modelu
# set_engine(num.threads = parallel::detectCores() - 1,)

# Jakub - decision_tree()
# Daria - random_forest()
# Maria - cubist_rules()
# Mateusz - xgboost()


