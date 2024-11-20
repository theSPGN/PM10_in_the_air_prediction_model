#loading data
load("prepared_data.RData")
#working on train_data, test_data, val_set, other_station_data, data_split

# creating cubist_rules model
library(rules)
library(Cubist)
library(tidymodels)


# cubist rules combines methods of decision trees and regression models
cr_mod <- cubist_rules(
  mode = "regression", #the only one possible
  committees = tune(),
  neighbors = tune(),
  max_rules = tune(),
  engine = "Cubist"
)

#making recipe
#1
cr1_recipe <- recipe(grimm_pm10 ~., data = train_data) |>
                  step_pca(all_numeric_predictors(), num_comp = 4) |>
                  update_role(date, new_role = "ID")
#2
cr2_recipe <- recipe(grimm_pm10 ~n_0750+ws+pres, data = train_data)

#workflows
workflow_1 <- workflow() |>
  add_recipe(cr1_recipe)|>
  add_model(cr_mod)

workflow_2 <- workflow() |>
  add_recipe(cr2_recipe)|>
  add_model(cr_mod)
#building grid
grid <- grid_regular(
  committees(range = c(1, 10)),
  neighbors(range = c(0,9)),
  max_rules(range = c(1,10)),
  levels =5
)

#tuning
set.seed(666)
cr1_res <- 
  workflow_1 |> 
  tune_grid(resamples = val_set, 
            grid = grid,
            control = control_grid(save_pred = T),
            metrics = metric_set(rmse, rsq, mae))

