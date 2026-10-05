################################################################################
# Save Trained Model for Shiny App
# Run this script after training your models to prepare for deployment
################################################################################

library(tidymodels)
library(readr)

# This script should be run at the end of your main analysis
# It saves the necessary objects for the Shiny app

################################################################################
# 1. Save the Best Model (LightGBM or whichever performed best)
################################################################################

# From your analysis, you have: best_model_fit
# Save it:
saveRDS(best_model_fit, "deployment/lightgbm_model.rds")

cat("✓ Model saved to: deployment/lightgbm_model.rds\n")

################################################################################
# 2. Save the Recipe (Preprocessing Pipeline)
################################################################################

# From your analysis, you have: ml_recipe
# Save it:
saveRDS(ml_recipe, "deployment/ml_recipe.rds")

cat("✓ Recipe saved to: deployment/ml_recipe.rds\n")

################################################################################
# 3. Save Model Performance Metrics
################################################################################

# Extract test set performance
test_metrics <- list(
  model_name = best_model_name,
  cv_rmse = best_model_row$CV_RMSE,
  cv_mae = best_model_row$CV_MAE,
  cv_r2 = best_model_row$CV_R2,
  test_rmse = rmse_tl,
  test_mae = mae_tl,
  test_r2 = r2_test
)

saveRDS(test_metrics, "deployment/model_metrics.rds")

cat("✓ Metrics saved to: deployment/model_metrics.rds\n")

################################################################################
# 4. Save Feature Importance
################################################################################

# Extract variable importance from LightGBM
lgbm_engine <- extract_fit_parsnip(best_model_fit)$fit

importance_df <- vip::vi(lgbm_engine) %>%
  arrange(desc(Importance)) %>%
  slice_head(n = 20)

saveRDS(importance_df, "deployment/feature_importance.rds")

cat("✓ Feature importance saved to: deployment/feature_importance.rds\n")

################################################################################
# 5. Create a Simple Prediction Function
################################################################################

# This function can be sourced by the Shiny app
predict_income <- function(new_data, model_path = "deployment/lightgbm_model.rds") {
  
  # Load model
  model <- readRDS(model_path)
  
  # Make prediction (log scale)
  pred_log <- predict(model, new_data = new_data)$.pred
  
  # Back-transform to original scale (TL)
  pred_income <- exp(pred_log)
  
  # Calculate confidence interval (approximate ±1.96 * residual std)
  # You can make this more sophisticated by storing residual distribution
  residual_sd <- 0.15  # Placeholder - update with actual residual SD from model
  
  lower_bound <- exp(pred_log - 1.96 * residual_sd)
  upper_bound <- exp(pred_log + 1.96 * residual_sd)
  
  # Return results
  data.frame(
    Predicted_Income = pred_income,
    Lower_95CI = lower_bound,
    Upper_95CI = upper_bound,
    Log_Prediction = pred_log
  )
}

# Save the function
saveRDS(predict_income, "deployment/predict_income_function.rds")

cat("✓ Prediction function saved to: deployment/predict_income_function.rds\n")

################################################################################
# 6. Create Sample Input Data for Testing
################################################################################

sample_customer <- data.frame(
  HF_CUSTOMER_AGE = 35,
  KKB_SCORE = 1200,
  KKB_CC_OPEN_TOT_LIM_AMT = 50000,
  KKB_CC_UTIL_RATIO = 0.5,
  KKB_CUST_TENURE = 24,
  HF_Odeyebilecegi_Taksit_Tutari = 3000,
  HF_AVG_ASSET_CURR_ACC_L3M = 5000,
  HF_ASSET_PRTCPATION_ACC_L3M = 1000,
  KKB_TOT_APP_CNT_L12M = 2,
  KKB_CC_OPEN_MAX_LIM_AMT = 30000,
  KKB_CL_TOT_RISK_AMT = 10000,
  KKB_MORTG_TOT_RISK_AMT = 0,
  KKB_CC_TOT_MONTHLY_PAYM_AMT = 1500,
  KKB_NON_DELQ_ACCT_CNT = 3,
  KKB_CC_OPEN_CNT = 2,
  KKB_CC_ACCT_CNT = 2,
  KKB_CURR_DELQ_DAY_CNT = 0,
  KKB_CURR_TOT_FOLLOW_AMT = 0,
  KKB_DELQ_ACCT_CNT = 0,
  HF_EDUCATION = "Üniversite",
  HF_APPLICANT_EMPLOYMENT_STATUS = "2",
  `İkamet İl` = "İSTANBUL",
  Tier = "T3",
  kira_il_ortalama = 8000
)

write.csv(sample_customer, "deployment/sample_customer.csv", row.names = FALSE)

cat("✓ Sample data saved to: deployment/sample_customer.csv\n")

################################################################################
# 7. Test the Prediction Pipeline
################################################################################

cat("\n--- Testing Prediction Pipeline ---\n")

# Load model and make test prediction
test_model <- readRDS("deployment/lightgbm_model.rds")
test_prediction <- predict(test_model, new_data = sample_customer)

cat("Test prediction (log scale):", test_prediction$.pred, "\n")
cat("Test prediction (TL):", exp(test_prediction$.pred), "\n")

cat("\n✓ All deployment files created successfully!\n")
cat("\nFiles created in 'deployment/' directory:\n")
cat("  - lightgbm_model.rds\n")
cat("  - ml_recipe.rds\n")
cat("  - model_metrics.rds\n")
cat("  - feature_importance.rds\n")
cat("  - predict_income_function.rds\n")
cat("  - sample_customer.csv\n")

cat("\nNext steps:\n")
cat("1. Copy 'income_prediction_app.R' to your deployment directory\n")
cat("2. Update the app to load these RDS files\n")
cat("3. Run: shiny::runApp('income_prediction_app.R')\n")
