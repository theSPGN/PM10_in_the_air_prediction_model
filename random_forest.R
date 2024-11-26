# Load necessary libraries
library(tidymodels)
library(dplyr)
library(ranger)
library(yardstick)
library(vip)
library(recipes)
library(corrplot)

# Set seed for reproducibility
set.seed(213)

# Load pre-prepared data
load("prepared_data.RData")

# Preprocessing

# Dodanie wizualizacji korelacji
# Oblicz macierz korelacji dla zmiennych numerycznych
cor_matrix <- cor(select_if(train_data, is.numeric), use = "complete.obs")

# Wykres korelacji z corrplot
corrplot(
    cor_matrix,
    method = "color", # styl wykresu
    col = colorRampPalette(c("#d73027", "white", "#1a9850"))(200),
    type = "upper", # rysuje tylko górny trójkąt
    addCoef.col = "black", # dodaje wartości korelacji
    tl.cex = 0.8, # wielkość tekstu etykiet
    number.cex = 0.7, # wielkość tekstu wartości
    title = "Macierz korelacji",
    mar = c(0, 0, 1, 0)
)

# brakuje ustawienia id_variables jako ID: funkcja update_role
# Remove low-correlation variables
id_variables <- train_data |>
    select_if(is.numeric) |>
    cor(use = "complete.obs") |>
    as.data.frame() |>
    select(grimm_pm10) |>
    abs() |>
    filter(grimm_pm10 < 0.3) |>
    rownames()

# Define the recipe
rf_recipe <- recipe(
    grimm_pm10 ~ .,
    data = train_data
) |>
    update_role(all_of(id_variables), new_role = "ID") |>
    step_naomit() |>
    step_time(date, features = c("hour")) |>
    step_rm(all_of(id_variables)) |>
    step_zv() |>
    step_normalize(all_numeric_predictors()) |>
    step_rm(date) |>
    step_corr(all_numeric_predictors(), threshold = 0.8) # |> # corr or pca?
# step_pca(all_numeric_predictors(), threshold = 0.9)

# Summary of recipe
rf_recipe |> summary()

# Dopasowanie przepisu do danych
pca_prep <- rf_recipe |> prep(training = train_data)

# Wyciągnięcie zmiennych po PCA
pca_vars <- tidy(pca_prep, number = 5) # Zakładamy, że `step_pca` jest piątym krokiem

# Podsumowanie
pca_removed <- pca_vars |>
    filter(terms != "") |> # Filtrujemy tylko istotne składowe
    select(component, terms) # Składowa i odpowiadające zmienne

# Wyświetlenie zmiennych powiązanych z każdą składową
print(pca_removed)


# Model
rf_model <- rand_forest(
    mode = "regression",
    mtry = tune(),
    min_n = tune(),
    trees = tune()
) |>
    set_engine("ranger",
        importance = "impurity",
        num.threads = parallel::detectCores() - 1
    )

# Workflow
rf_workflow <- workflow() |>
    add_recipe(rf_recipe) |>
    add_model(rf_model)

# Hyperparameter tuning grid
rf_grid <- grid_regular(
    mtry(range = c(2, 10)),
    min_n(range = c(1, 10)),
    trees(range = c(100, 300)),
    levels = 5
)

# Control settings to display progress and collect metrics
control <- control_grid(verbose = TRUE, save_pred = TRUE, parallel_over = "everything")

# Cross-validation, Stratify to balance distribution
cv_folds <- vfold_cv(train_data, v = 5, strata = grimm_pm10)

rf_res <- rf_workflow |>
    tune_grid(
        resamples = cv_folds,
        grid = rf_grid,
        control = control
    )

# Select the best model based on RMSE
best_rf <- rf_res |>
    select_best(metric = "rmse")

# Finalize the workflow with the best hyperparameters
rf_fit <- rf_workflow |>
    finalize_workflow(best_rf) |>
    fit(data = train_data)

# Generate and visualize variable importance
vip(rf_fit$fit$fit, num_features = 10)

# Save the final model
save(rf_fit, file = "rf_fit.RData")

# Predictions on test data
predictions <- rf_fit |>
    predict(new_data = test_data) |>
    bind_cols(test_data)

# Evaluate the model performance
model_metrics <- predictions |>
    metrics(truth = grimm_pm10, estimate = .pred)

print(model_metrics)

# Plot predictions vs truth
predictions |>
    ggplot(aes(x = grimm_pm10, y = .pred)) +
    geom_point(alpha = 0.6, color = "#00a2ff") +
    geom_abline(slope = 1, intercept = 0, color = "#63438b", linetype = "dashed") +
    labs(
        title = "Predictions vs. Truth",
        x = "Truth (Actual Values)",
        y = "Predictions"
    ) +
    geom_abline(slope = 0.5, intercept = 0, color = "#63438b", linetype = "dashed") +
    labs(
        title = "Predictions vs. Truth",
        x = "Truth (Actual Values)",
        y = "Predictions"
    ) +
    geom_abline(slope = 2, intercept = 0, color = "#63438b", linetype = "dashed") +
    labs(
        title = "Predictions vs. Truth",
        x = "Truth (Actual Values)",
        y = "Predictions"
    ) +
    theme_minimal()

# Save the predictions and workflow
save(predictions, file = "predictions.RData")
save(rf_workflow, file = "rf_workflow.RData")


# PCA
#   <chr>   <chr>          <dbl>
# 1 rmse    standard       6.40
# 2 rsq     standard       0.964
# 3 mae     standard       4.58

# Corr
#   .metric .estimator .estimate
#   <chr>   <chr>          <dbl>
# 1 rmse    standard       6.30
# 2 rsq     standard       0.965
# 3 mae     standard       4.73
