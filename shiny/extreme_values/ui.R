library(shiny)

shinyUI(
  fluidPage(
    titlePanel("IE 477 – Effect of Extreme Values on Linear Fit"),
    
    sidebarLayout(
      sidebarPanel(
        selectInput(
          "x_var",
          "Predictor on x-axis:",
          choices  = c(
            "Age"                           = "HF_CUSTOMER_AGE",
            "KKB Score"                     = "KKB_SCORE",
            "Credit Card Utilization"       = "KKB_CC_UTIL_RATIO",
            "Total Credit Card Limit"       = "KKB_CC_OPEN_TOT_LIM_AMT",
            "Average Assets (Curr Acc, L3M)"= "HF_AVG_ASSET_CURR_ACC_L3M"
          ),
          selected = "HF_CUSTOMER_AGE"
        ),
        sliderInput(
          "quantile_cut",
          "Trim top income quantile:",
          min   = 0.90,
          max   = 0.999,
          value = 0.98,
          step  = 0.001
        ),
        checkboxInput(
          "log_scale",
          "Show income on log-scale (y-axis)",
          value = TRUE
        ),
        helpText("Red points = trimmed high-income observations (above cutoff).")
      ),
      
      mainPanel(
        plotOutput("regPlot",  height = "450px"),
        plotOutput("diagPlot", height = "350px")
      )
    )
  )
)

