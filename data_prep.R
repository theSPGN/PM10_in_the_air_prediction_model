library(tidymodels)
library(parsnip)
library(openair)
library(ggplot2)
library(GGally)
library(factoextra)

# Wczytanie danych do trenowania modelu
load("ops.RData")
ops <- ops |> na.omit()

# Wczytanie danych z zewnętrznej stacji
# nie jest to zbiór testowy
load("data_test.rdata")

# Wykresy korelacji danych
# ops |> select(-date, - ops_pm10) |> ggpairs()

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

#dane do testowania
model_station_data <- ops |> select(-pres_sea, -ops_pm10)

#małe pca
pca_result <- prcomp(model_station_data[,-1], center = TRUE, scale. = TRUE)
summary(pca_result)
pca_result$rotation
fviz_eig(pca_result, addlabels = TRUE)
# update_role(rh, temp, wd, prec, new_role="ID") bo są słabo skorelowane lub za bardzo

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

ops_data <- ops_data |>
  select(-grimm_pm10) |>
  left_join(bam, by = "date") |>
  rename(grimm_pm10 = bam_pm10) |>
  na.omit()

other_station_data <-
  ops_data |>
  select(colnames(model_station_data))


rm(list = c("ops", "bam", "ops_bam", "ops_data", "model_station_data"))

save(train_data, test_data, val_set, other_station_data, data_split, file = "prepared_data.RData")

# Notatki
# output grimm_pm10
# dobra korelacja cząstek n0200-n1000 oraz wind speed
# resample = val_set
# set_engine(num.threads = parallel::detectCores() - 1,)

# Jakub - decision_tree()
# Daria - random_forest()
# Maria - cubist_rules()
# Mateusz - xgboost()