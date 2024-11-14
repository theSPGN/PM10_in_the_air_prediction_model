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
  step_pca(all_numeric(), threshold = 0.95) |> 
  step_dummy(all_nominal()) 

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

# Ustawienie walidacji krzyżowej dla trenowania
set.seed(123)
folds <- vfold_cv(train_data, v = 5, strata = grimm_pm10)

# Strojenie parametrów przy użyciu siatki i walidacji krzyżowej
tune_results <- tune_grid(
  workflow_tree,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(rmse, rsq)   
)

# Wyświetlenie najlepszych wyników
show_best(tune_results, "rmse")

# Wybór najlepszej kombinacji parametrów na podstawie RMSE
best_params <- select_best(tune_results, "rmse")

# Finalizacja workflow z najlepszymi parametrami
final_workflow <- finalize_workflow(workflow_tree, best_params)

# Dopasowanie ostatecznego modelu do całego zbioru treningowego
final_fit <- fit(final_workflow, data = train_data)

# Ocena ostatecznego modelu na zbiorze testowym
final_results <- last_fit(final_workflow, split = data_split)

# Wyświetlenie końcowej wydajności na zbiorze testowym
collect_metrics(final_results)