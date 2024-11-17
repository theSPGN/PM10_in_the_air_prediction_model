# Load necessary libraries
library(tidymodels)
library(dplyr)
library(ranger)
library(yardstick)
library(vip)
library(recipes)

# Set seed for reproducibility
set.seed(213)

# Load pre-prepared data
load("prepared_data.RData")

# Preprocessing

# Remove zero-variance columns
zv_cols <- sapply(train_data, function(x) length(unique(x)) == 1)
train_data <- train_data[, !zv_cols]

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
    step_rm(all_of(id_variables)) |>
    step_normalize(all_numeric_predictors()) |>
    step_corr(all_numeric_predictors(), threshold = 0.8) |>
    step_pca(all_numeric_predictors(), threshold = 0.9)

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
# # A tibble: 3 × 3
#   .metric .estimator .estimate
#   <chr>   <chr>          <dbl>
# 1 rmse    standard       6.30
# 2 rsq     standard       0.965
# 3 mae     standard       4.73

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
    theme_minimal()

# Save the predictions and workflow
save(predictions, file = "predictions.RData")
save(rf_workflow, file = "rf_workflow.RData")
