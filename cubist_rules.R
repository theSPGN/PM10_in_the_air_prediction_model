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

#3
cr3_recipe <- recipe(grimm_pm10 ~., data = train_data) |>
  step_pca(all_numeric_predictors(), num_comp = 5) |>
  update_role(date, new_role = "ID")
#4
cr4_recipe <- recipe(grimm_pm10 ~n_0250+n_0044+mws+pres, data = train_data)

#workflows
workflow_1 <- workflow() |>
  add_recipe(cr1_recipe)|>
  add_model(cr_mod)

workflow_2 <- workflow() |>
  add_recipe(cr2_recipe)|>
  add_model(cr_mod)

workflow_3 <- workflow() |>
  add_recipe(cr3_recipe)|>
  add_model(cr_mod)

workflow_4 <- workflow() |>
  add_recipe(cr4_recipe)|>
  add_model(cr_mod)

#building grid
grid <- grid_regular(
  committees(range = c(1,50)),
  neighbors(range = c(0,9)),
  max_rules(range = c(1,15)),
  levels =9
)

#tuning
set.seed(666)
cr1_res <- 
  workflow_1 |> 
  tune_grid(resamples = val_set, 
            grid = grid,
            control = control_grid(save_pred = T),
            metrics = metric_set(rmse, rsq, mae))
cr2_res <- 
  workflow_2 |> 
  tune_grid(resamples = val_set, 
            grid = grid,
            control = control_grid(save_pred = T),
            metrics = metric_set(rmse, rsq, mae))

cr3_res <- 
  workflow_3 |> 
  tune_grid(resamples = val_set, 
            grid = grid,
            control = control_grid(save_pred = T),
            metrics = metric_set(rmse, rsq, mae))

cr4_res <- 
  workflow_4 |> 
  tune_grid(resamples = val_set, 
            grid = grid,
            control = control_grid(save_pred = T),
            metrics = metric_set(rmse, rsq, mae))

#selecting best parameters
cr1_b<- select_best(cr1_res)
cr2_b<- select_best(cr2_res)
cr3_b<- select_best(cr3_res)
cr4_b<- select_best(cr4_res)

#collecting predictions
cr1_fit<-
  cr1_res|> collect_predictions(parameters = cr1_b)|>
  mutate(results = "cr1")
print(metrics(cr1_fit, truth = grimm_pm10, estimate = .pred))

cr2_fit<-
  cr2_res|> collect_predictions(parameters = cr2_b)|>
  mutate(results = "cr2")
print(metrics(cr2_fit, truth = grimm_pm10, estimate = .pred))

cr3_fit<-
  cr3_res|> collect_predictions(parameters = cr3_b)|>
  mutate(results = "cr3")
print(metrics(cr3_fit, truth = grimm_pm10, estimate = .pred))

#the best one
cr4_fit<-
  cr4_res|> collect_predictions(parameters = cr4_b)|>
  mutate(results = "cr4")
print(metrics(cr4_fit, truth = grimm_pm10, estimate = .pred))

#plotting two best model against actual data
bind_rows(cr1_fit,cr4_fit) |> 
  ggplot(aes(x = grimm_pm10, y = .pred,  col = results), col = results) +
  geom_abline(slope = 1, intercept = 0, color = "black",linewidth=1.5, alpha = 0.5)+
  geom_point(size=1)+
  geom_smooth()+
  labs(title = "Grimm_pm10 vs. predicted values", x = "Grimm_pm10", y = "Prediction")

# saving last fits
save(cr4_recipe,workflow_4, cr4_res, cr4_b, cr4_fit, file = "cubist_rules_best_fit.RData")

# Final fits
cr1_bmwf <- workflow_1 |>
  finalize_workflow(cr1_b)|>
  last_fit(split=data_split)

print(cr1_bmwf$.metrics)
# A tibble: 2 × 4
# .metric .estimator .estimate .config             
# <chr>   <chr>          <dbl> <chr>               
#  1 rmse    standard      17.2   Preprocessor1_Model1
#  2 rsq     standard       0.711 Preprocessor1_Model1
cr4_bmwf <- workflow_4 |>
  finalize_workflow(cr4_b) |>
  last_fit(split=data_split)

print(cr4_bmwf$.metrics)
# A tibble: 2 × 4
#.metric .estimator .estimate .config             
#<chr>   <chr>          <dbl> <chr>               
#  1 rmse    standard       9.57  Preprocessor1_Model1
#  2 rsq     standard       0.914 Preprocessor1_Model1
