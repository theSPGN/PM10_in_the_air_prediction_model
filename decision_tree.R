library(tidymodels)
library(recipes)
library(DT)

load("prepared_data.RData")

# Wyświetlenie danych
datatable(train_data, 
          options = list(pageLength = 50, 
                         autoWidth = TRUE, 
                         scrollX = TRUE))

recipe_tree <- recipe(grimm_pm10 ~ ., data = train_data)  |> 
  step_rm(date) %>%
  step_date(all_nominal(), features = c("dow", "month")) |> 
  step_time(all_nominal(), features = c("hour")) |> 
  step_corr(all_numeric(), threshold = 0.85) |> 
  step_normalize(all_numeric(), -all_outcomes()) |> 
  step_nzv(all_predictors()) |> 
  step_pca(all_numeric(), threshold = 0.95)


# Sprawdzenie receptury
prepped_recipe <- prep(recipe_tree, training = train_data)
baked_data <- bake(prepped_recipe, new_data = NULL)