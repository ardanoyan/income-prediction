library(shiny)
library(tidyverse)
library(readxl)


df_raw <- read_excel(
  "data/synthetic_customers.xlsx"
)

model_data_scaled <- df_raw %>%
  mutate(
    Edevlet_Net_Gelir         = as.numeric(Edevlet_Net_Gelir),
    HF_CUSTOMER_AGE           = as.numeric(HF_CUSTOMER_AGE),
    KKB_SCORE                 = as.numeric(KKB_SCORE),
    KKB_CC_UTIL_RATIO         = as.numeric(KKB_CC_UTIL_RATIO),
    KKB_CC_OPEN_TOT_LIM_AMT   = as.numeric(KKB_CC_OPEN_TOT_LIM_AMT),
    HF_AVG_ASSET_CURR_ACC_L3M = as.numeric(HF_AVG_ASSET_CURR_ACC_L3M)
  ) %>%
  filter(Edevlet_Net_Gelir > 0) %>%
  mutate(
    log_income = log(Edevlet_Net_Gelir)
  ) %>%
  select(
    HF_CUSTOMER_AGE,
    KKB_SCORE,
    KKB_CC_UTIL_RATIO,
    KKB_CC_OPEN_TOT_LIM_AMT,
    HF_AVG_ASSET_CURR_ACC_L3M,
    log_income
  ) %>%
  drop_na()

# UI'daki isimleri düzgün göstermek için label map
choices_vec <- c(
  "Age"                           = "HF_CUSTOMER_AGE",
  "KKB Score"                     = "KKB_SCORE",
  "Credit Card Utilization"       = "KKB_CC_UTIL_RATIO",
  "Total Credit Card Limit"       = "KKB_CC_OPEN_TOT_LIM_AMT",
  "Average Assets (Curr Acc, L3M)"= "HF_AVG_ASSET_CURR_ACC_L3M"
)
label_map <- setNames(names(choices_vec), choices_vec)


# ===== SERVER =============================================================

shinyServer(function(input, output, session) {
  

  base_data <- reactive({
    req(input$x_var)
    model_data_scaled %>%
      transmute(
        x          = .data[[input$x_var]],
        log_income = log_income
      ) %>%
      drop_na()
  })
  

  cutoff_val <- reactive({
    quantile(base_data()$log_income, input$quantile_cut, na.rm = TRUE)
  })
  

  trimmed_data <- reactive({
    base_data() %>%
      filter(log_income <= cutoff_val())
  })
  
  # Lineer modeller
  lm_all  <- reactive({ lm(log_income ~ x, data = base_data()) })
  lm_trim <- reactive({ lm(log_income ~ x, data = trimmed_data()) })
  

  output$regPlot <- renderPlot({
    d_all   <- base_data()
    d_trim  <- trimmed_data()
    cut_val <- cutoff_val()
    
    x_label <- label_map[[input$x_var]]
    

    x_grid <- data.frame(
      x = seq(
        from = min(d_all$x, na.rm = TRUE),
        to   = max(d_all$x, na.rm = TRUE),
        length.out = 200
      )
    )
    
    pred_all  <- predict(lm_all(),  newdata = x_grid)
    pred_trim <- predict(lm_trim(), newdata = x_grid)
    
    plot_df <- x_grid %>%
      mutate(
        pred_all  = pred_all,
        pred_trim = pred_trim
      )
    
    ggplot() +
      geom_point(
        data  = d_all,
        aes(x = x, y = log_income),
        alpha = 0.25,
        color = "grey40"
      ) +
      geom_point(
        data  = d_all %>% filter(log_income > cut_val),
        aes(x = x, y = log_income),
        color = "red",
        size  = 2
      ) +
      geom_line(
        data = plot_df,
        aes(x = x, y = pred_all, color = "All data"),
        linewidth = 1.1
      ) +
      geom_line(
        data = plot_df,
        aes(x = x, y = pred_trim, color = "Trimmed (below cutoff)"),
        linewidth = 1.1,
        linetype  = "dashed"
      ) +
      scale_color_manual(
        name   = "Regression fit",
        values = c(
          "All data"               = "black",
          "Trimmed (below cutoff)" = "dodgerblue3"
        )
      ) +
      labs(
        title = "Effect of Extreme Incomes on Linear Fit",
        subtitle = paste0(
          "Predictor: ", x_label,
          " | Cutoff: top ",
          round((1 - input$quantile_cut) * 100, 3),
          "% of log-income trimmed (red points)"
        ),
        x  = x_label,
        y  = "Log-Income"
      ) +
      theme_minimal(base_size = 14)
  })
  

  output$diagPlot <- renderPlot({
    par(mfrow = c(1, 2), mar = c(5, 5, 3, 2))
    
    plot(
      lm_all(),
      which = 1,
      main  = "Residuals vs Fitted (All Data)"
    )
    
    plot(
      lm_trim(),
      which = 1,
      main  = "Residuals vs Fitted (Trimmed Data)"
    )
    
    par(mfrow = c(1, 1))
  })
})
