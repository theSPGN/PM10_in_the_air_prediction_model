library(tidymodels)
library(recipes)
library(DT)

# Wczytanie przetworzonych danych
load("prepared_data.RData")

# Wyświetlenie danych
# datatable(train_data, 
#           options = list(pageLength = 50, 
#                          autoWidth = TRUE, 
#                          scrollX = TRUE))

recipe_tree <- recipe(grimm_pm10 ~ ., data = train_data)  |> 
  step_rm(date) |> 
  step_date(all_nominal(), features = c("dow", "month")) |> 
  step_time(all_nominal(), features = c("hour")) |> 
  step_corr(all_numeric(), threshold = 0.85) |> 
  step_normalize(all_numeric(), -all_outcomes()) |> 
  step_nzv(all_predictors()) |> 
  step_pca(all_numeric(), threshold = 0.95)

# Sprawdzenie receptury
prepped_recipe <- prep(recipe_tree, training = train_data)
baked_data <- bake(prepped_recipe, new_data = NULL)

head(baked_data)

# Wyświetlenie przetworzonych danych
datatable(baked_data, 
          options = list(pageLength = 50, 
                         autoWidth = TRUE, 
                         scrollX = TRUE))

# Definiowanie modelu drzewa decyzyjnego z parametrami do tuningu
tree_spec <- decision_tree(
  cost_complexity = tune(),  
  tree_depth = tune(),       
  min_n = tune()
) |> 
  set_mode("regression") |> 
  set_engine("rpart")

# Tworzenie workflow z recepturą i specyfikacją modelu
workflow_tree <- workflow() |> 
  add_recipe(recipe_tree) |> 
  add_model(tree_spec)

# Definiowanie siatki możliwych wartości - wstępnie
tree_grid <- grid_regular(
  cost_complexity(range = c(-3, -1)), 
  tree_depth(range = c(5, 15)),
  min_n(range = c(2, 10)),        
  levels = 5
)