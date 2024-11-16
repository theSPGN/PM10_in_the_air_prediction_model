library(tidymodels)
library(recipes)
library(DT)

# Wczytanie przetworzonych danych
load("prepared_data.RData")

# Wyświetlenie danych
datatable(train_data, 
          options = list(pageLength = 50, 
                         autoWidth = TRUE, 
                         scrollX = TRUE))

# Definiowanie receptury
recipe_tree <- recipe(grimm_pm10 ~ ., data = train_data) |>
  step_rm(date) |> 
  step_date(all_nominal(), features = c("dow", "month")) |> 
  step_time(all_nominal(), features = c("hour")) |> 
  step_corr(all_numeric(), -all_outcomes(), threshold = 0.85) |> 
  step_normalize(all_numeric(), -all_outcomes()) |> 
  step_nzv(all_predictors()) |> 
  step_pca(all_numeric(), -all_outcomes(), threshold = 0.95) |> 
  step_dummy(all_nominal())

# Sprawdzenie receptury
prepped_recipe <- prep(recipe_tree, training = train_data)
baked_data <- bake(prepped_recipe, new_data = NULL)

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

# Definiowanie siatki możliwych wartości
tree_grid <- grid_regular(
  cost_complexity(range = c(-3, -1)), 
  tree_depth(range = c(5, 15)),
  min_n(range = c(2, 10)),        
  levels = 10
)

# Ustawienie walidacji krzyżowej dla trenowania
set.seed(456)
folds <- vfold_cv(train_data, v = 10, strata = grimm_pm10)

# Strojenie parametrów przy użyciu siatki i walidacji krzyżowej
tune_results <- tune_grid(
  workflow_tree,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(rmse, rsq, mae) # Dodano MAE
)

# Wyświetlenie najlepszych wyników
show_best(tune_results, metric = "rmse")

# Wybór najlepszej kombinacji parametrów na podstawie RMSE
best_params <- select_best(tune_results, metric = "rmse")

# Finalizacja workflow z najlepszymi parametrami
final_workflow <- finalize_workflow(workflow_tree, best_params)

# Dopasowanie ostatecznego modelu do całego zbioru treningowego
final_fit <- fit(final_workflow, data = train_data)

# Ocena ostatecznego modelu na zbiorze testowym
final_results <- last_fit(final_workflow, split = data_split)

# Wyświetlenie końcowej wydajności na zbiorze testowym
collect_metrics(final_results)

# Zapisanie modelu i wyników do pliku .RData
save(final_results, file = "last_fit_decision_tree.RData")


# dla level 5 oraz 5 folds
# .metric .estimator .estimate .config             
# <chr>   <chr>          <dbl> <chr>               
#   1 rmse    standard      14.7   Preprocessor1_Model1
# 2 rsq     standard       0.848 Preprocessor1_Model1

# dla level 10 oraz 10 folds
# .metric .estimator .estimate .config             
# <chr>   <chr>          <dbl> <chr>               
#   1 rmse    standard      14.7   Preprocessor1_Model1
# 2 rsq     standard       0.847 Preprocessor1_Model1


# dla level 10 oraz 10 folds Zamiast val_set używany jest vfold_cv() z 10 grupami do walidacji krzyżowej.
# .metric .estimator .estimate .config             
# <chr>   <chr>          <dbl> <chr>               
#   1 rmse    standard      14.7   Preprocessor1_Model1
# 2 rsq     standard       0.847 Preprocessor1_Model1


# Najniższe wartości RMSE otrzymano dla algorytmu X" Wartość RMSE wynosiła. Najwyższe wartość RMSE otrzymano dla ... . Różnica w wartośc RMSE była ... jaka ?
# wnioski dodam jak przeprowadzę jeszcze kilka eksperymentów
