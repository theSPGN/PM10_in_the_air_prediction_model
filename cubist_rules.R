#loading data
load("prepared_data.RData")
#working on train_data, test_data, val_set, other_station_data, data_split

# creating cubist_rules model

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
cr1_recipe <- recipe(grimm_pm10 ~., data = train_data,
                  step_pca(all_numeric_predictors(), num_comp = 4),
                  update_role(date, new_role = "ID"))
#2
cr2_recipe <- recipe(grimm_pm10 ~n_0750+ws+pres, data = train_data)
