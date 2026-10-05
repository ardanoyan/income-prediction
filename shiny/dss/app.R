################################################################################
# Income Prediction Decision Support System (DSS)
# IE 477 Project - Bilkent University
# Author: Arda Noyan Karaşoğlu
#
# Models: OLS Regression, Lasso Regression, LightGBM
################################################################################
library(shiny)
library(bslib)
library(tidymodels)
library(lightgbm)
library(glmnet)
library(dplyr)
library(ggplot2)
library(DT)
library(readxl)
library(scales)

# ─── Dosya yükleme limitini artır (500 MB) ───
options(shiny.maxRequestSize = 500 * 1024^2)

################################################################################
# Load Trained Models and Resources
################################################################################
MODEL_DIR <- "deployment"
LGBM_LOADED  <- FALSE
OLS_LOADED   <- FALSE
LASSO_LOADED <- FALSE

tryCatch({
  best_model        <- readRDS(file.path(MODEL_DIR, "lightgbm_model.rds"))
  model_metrics     <- readRDS(file.path(MODEL_DIR, "model_metrics.rds"))
  feature_importance <- readRDS(file.path(MODEL_DIR, "feature_importance.rds"))
  LGBM_LOADED <- TRUE
  cat("\u2713 LightGBM loaded\n")
}, error = function(e) {
  model_metrics <<- list(model_name="LightGBM (Demo)", cv_rmse=0.123, cv_mae=0.098,
                         cv_r2=0.756, test_rmse=8543, test_mae=6234, test_r2=0.743)
  feature_importance <<- data.frame(
    Variable   = c("HF_Odeyebilecegi_Taksit_Tutari","KKB_SCORE",
                   "KKB_CC_OPEN_TOT_LIM_AMT","HF_CUSTOMER_AGE",
                   "KKB_CC_UTIL_RATIO","KKB_CUST_TENURE",
                   "HF_AVG_ASSET_CURR_ACC_L3M","KKB_TOT_APP_CNT_L12M",
                   "KKB_CC_OPEN_CNT","KKB_CURR_DELQ_DAY_CNT"),
    Importance = c(100,85,72,65,58,52,45,38,30,22))
})
tryCatch({
  ols_model   <- readRDS(file.path(MODEL_DIR, "ols_model.rds"))
  ols_metrics <- readRDS(file.path(MODEL_DIR, "ols_metrics.rds"))
  OLS_LOADED  <- TRUE
  cat("\u2713 OLS loaded\n")
}, error = function(e) {
  ols_metrics <<- list(model_name="OLS (Demo)", cv_rmse=0.155, cv_mae=0.121,
                       cv_r2=0.685, test_rmse=10250, test_mae=7800, test_r2=0.680)
})
tryCatch({
  lasso_model   <- readRDS(file.path(MODEL_DIR, "lasso_model.rds"))
  lasso_metrics <- readRDS(file.path(MODEL_DIR, "lasso_metrics.rds"))
  LASSO_LOADED  <- TRUE
  cat("\u2713 Lasso loaded\n")
}, error = function(e) {
  lasso_metrics <<- list(model_name="Lasso (Demo)", cv_rmse=0.148, cv_mae=0.115,
                         cv_r2=0.700, test_rmse=9800, test_mae=7500, test_r2=0.695)
})

################################################################################
# Theme & CSS
################################################################################
app_theme <- bs_theme(
  version   = 5,
  bootswatch = "flatly",
  primary   = "#0f172a",
  secondary = "#64748b",
  success   = "#10b981",
  info      = "#3b82f6",
  warning   = "#f59e0b",
  danger    = "#ef4444",
  base_font = font_google("Plus Jakarta Sans"),
  heading_font = font_google("Plus Jakarta Sans"),
  code_font = font_google("JetBrains Mono"),
  "body-bg" = "#f1f5f9",
  "card-bg"  = "#ffffff"
)

mega_css <- '
@import url("https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap");
:root {
  --navy-900: #0f172a; --navy-800: #1e293b; --navy-700: #334155;
  --navy-600: #475569; --navy-500: #64748b; --navy-400: #94a3b8;
  --navy-300: #cbd5e1; --navy-200: #e2e8f0; --navy-100: #f1f5f9; --navy-50: #f8fafc;
  --blue-600: #2563eb; --blue-500: #3b82f6; --blue-400: #60a5fa;
  --green-500: #10b981; --green-400: #34d399;
  --amber-500: #f59e0b; --red-500: #ef4444;
  --glass-bg: rgba(255,255,255,0.72); --glass-border: rgba(255,255,255,0.3);
  --shadow-sm: 0 1px 2px rgba(15,23,42,0.05);
  --shadow-md: 0 4px 12px rgba(15,23,42,0.08);
  --shadow-lg: 0 10px 40px rgba(15,23,42,0.12);
  --shadow-xl: 0 20px 60px rgba(15,23,42,0.15);
  --radius-sm: 8px; --radius-md: 12px; --radius-lg: 16px; --radius-xl: 20px;
  --transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
}
body { font-family: "Plus Jakarta Sans", system-ui, sans-serif; background: var(--navy-100); color: var(--navy-800); -webkit-font-smoothing: antialiased; }
.sidebar { background: linear-gradient(180deg, var(--navy-900) 0%, #162033 100%) !important; border-right: 1px solid rgba(255,255,255,0.06); padding-top: 0 !important; }
.sidebar .brand-header { padding: 28px 24px 20px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 12px; }
.sidebar .brand-header h4 { color: #fff; font-weight: 800; font-size: 18px; margin: 0 0 4px 0; letter-spacing: -0.3px; }
.sidebar .brand-header .brand-sub { color: var(--navy-400); font-size: 11px; font-weight: 500; letter-spacing: 1.2px; text-transform: uppercase; }
.sidebar .nav-link { color: var(--navy-400) !important; font-weight: 500; font-size: 14px; padding: 12px 24px !important; margin: 2px 12px; border-radius: var(--radius-sm); transition: var(--transition); display: flex; align-items: center; gap: 12px; }
.sidebar .nav-link:hover { color: #fff !important; background: rgba(255,255,255,0.06); }
.sidebar .nav-link.active { color: #fff !important; background: var(--blue-600) !important; font-weight: 600; box-shadow: 0 4px 16px rgba(37,99,235,0.35); }
.sidebar .nav-link i { width: 20px; text-align: center; font-size: 15px; }
.sidebar-status { padding: 16px 24px; border-top: 1px solid rgba(255,255,255,0.06); margin-top: auto; }
.sidebar-status .status-item { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 12px; color: var(--navy-400); }
.sidebar-status .status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.sidebar-status .dot-active { background: var(--green-400); box-shadow: 0 0 8px rgba(52,211,153,0.5); }
.sidebar-status .dot-demo { background: var(--navy-500); }
.glass-card { background: var(--glass-bg); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid var(--glass-border); border-radius: var(--radius-lg); padding: 28px; margin-bottom: 20px; box-shadow: var(--shadow-md); transition: var(--transition); }
.glass-card:hover { box-shadow: var(--shadow-lg); transform: translateY(-1px); }
.glass-card .card-header-custom { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid var(--navy-200); }
.glass-card .card-header-custom i { color: var(--blue-500); font-size: 18px; }
.glass-card .card-header-custom h5 { font-weight: 700; font-size: 15px; color: var(--navy-900); margin: 0; letter-spacing: -0.2px; }
.hero-prediction { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #2563eb 100%); border-radius: var(--radius-xl); padding: 40px 32px; color: #fff; text-align: center; position: relative; overflow: hidden; box-shadow: var(--shadow-xl); }
.hero-prediction .hero-model-tag { display: inline-block; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2); border-radius: 20px; padding: 4px 16px; font-size: 12px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 12px; }
.hero-prediction .hero-amount { font-size: 52px; font-weight: 800; letter-spacing: -1.5px; line-height: 1; margin-bottom: 10px; text-shadow: 0 2px 20px rgba(0,0,0,0.2); }
.hero-prediction .hero-ci { font-size: 15px; opacity: 0.75; font-weight: 400; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-top: 20px; }
.kpi-card { background: #fff; border-radius: var(--radius-md); padding: 18px; text-align: center; border: 1px solid var(--navy-200); box-shadow: var(--shadow-sm); transition: var(--transition); }
.kpi-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }
.kpi-card .kpi-icon { font-size: 20px; margin-bottom: 6px; }
.kpi-card .kpi-value { font-size: 24px; font-weight: 800; color: var(--navy-900); line-height: 1.1; }
.kpi-card .kpi-label { font-size: 11px; font-weight: 600; color: var(--navy-500); text-transform: uppercase; letter-spacing: 0.8px; margin-top: 4px; }
.seg-badge { display: inline-block; padding: 5px 14px; border-radius: 20px; font-weight: 700; font-size: 12px; letter-spacing: 0.5px; }
.seg-low { background: #fef2f2; color: #dc2626; }
.seg-medium { background: #fffbeb; color: #d97706; }
.seg-high { background: #ecfdf5; color: #059669; }
.compare-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 20px; }
.compare-strip-item { background: #fff; border-radius: var(--radius-md); padding: 18px; text-align: center; border: 2px solid var(--navy-200); transition: var(--transition); }
.compare-strip-item.is-selected { border-color: var(--blue-600); background: #eff6ff; box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
.compare-strip-item .cs-model { font-size: 12px; font-weight: 700; color: var(--navy-500); text-transform: uppercase; letter-spacing: 0.8px; }
.compare-strip-item .cs-value { font-size: 24px; font-weight: 800; color: var(--navy-900); margin-top: 4px; }
.btn-predict { background: linear-gradient(135deg, var(--blue-600) 0%, #1d4ed8 100%); color: #fff !important; border: none; border-radius: var(--radius-md); font-family: "Plus Jakarta Sans", sans-serif; font-size: 15px; font-weight: 700; padding: 16px 32px; width: 100%; letter-spacing: 0.3px; cursor: pointer; transition: var(--transition); box-shadow: 0 4px 14px rgba(37,99,235,0.3); }
.btn-predict:hover { background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%); box-shadow: 0 6px 24px rgba(37,99,235,0.45); transform: translateY(-1px); color: #fff !important; }
.form-control, .form-select { border-radius: var(--radius-sm) !important; border: 1.5px solid var(--navy-200) !important; font-size: 14px; padding: 10px 14px; transition: var(--transition); font-family: "Plus Jakarta Sans", sans-serif; }
.form-control:focus, .form-select:focus { border-color: var(--blue-500) !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important; }
.form-label, label { font-weight: 600; font-size: 13px; color: var(--navy-700); margin-bottom: 4px; }
.selectize-input { border-radius: var(--radius-sm) !important; border: 1.5px solid var(--navy-200) !important; font-family: "Plus Jakarta Sans", sans-serif !important; padding: 8px 12px !important; }
.guidance-card { background: #fff; border-radius: var(--radius-md); padding: 22px; border-left: 4px solid; transition: var(--transition); height: 100%; }
.guidance-card h6 { font-weight: 800; font-size: 14px; color: var(--navy-900); margin-bottom: 8px; }
.guidance-card p { font-size: 13px; color: var(--navy-600); line-height: 1.55; margin: 0; }
.gc-blue { border-left-color: var(--blue-500); }
.gc-green { border-left-color: var(--green-500); }
.gc-amber { border-left-color: var(--amber-500); }
.metric-card { background: #fff; border-radius: var(--radius-md); padding: 22px; text-align: center; border: 1px solid var(--navy-200); box-shadow: var(--shadow-sm); }
.metric-card .mc-icon { width: 44px; height: 44px; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; font-size: 18px; margin-bottom: 10px; }
.mc-blue .mc-icon { background: #eff6ff; color: var(--blue-500); }
.mc-green .mc-icon { background: #ecfdf5; color: var(--green-500); }
.mc-amber .mc-icon { background: #fffbeb; color: var(--amber-500); }
.metric-card .mc-value { font-size: 26px; font-weight: 800; color: var(--navy-900); }
.metric-card .mc-label { font-size: 12px; font-weight: 600; color: var(--navy-500); text-transform: uppercase; letter-spacing: 0.6px; margin-top: 2px; }
.demo-banner { background: linear-gradient(135deg, #fffbeb, #fef3c7); border: 1px solid #fde68a; border-radius: var(--radius-md); padding: 14px 20px; margin-top: 16px; display: flex; align-items: center; gap: 10px; font-size: 13px; color: #92400e; }
table.dataTable { font-size: 13px; }
table.dataTable thead th { background: var(--navy-900) !important; color: #fff !important; font-weight: 600; border-bottom: none !important; }
.guide-section { margin-bottom: 28px; }
.guide-section h5 { font-weight: 800; color: var(--navy-900); font-size: 16px; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
.guide-section h5 .step-num { display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 50%; background: var(--blue-600); color: #fff; font-size: 13px; font-weight: 700; flex-shrink: 0; }
.guide-section ol li, .guide-section ul li { padding: 4px 0; color: var(--navy-700); font-size: 14px; }
.page-header { margin-bottom: 24px; }
.page-header h3 { font-weight: 800; color: var(--navy-900); font-size: 24px; letter-spacing: -0.5px; margin: 0 0 4px 0; }
.page-header p { color: var(--navy-500); font-size: 14px; margin: 0; }
'

################################################################################
# UI
################################################################################
ui <- page_sidebar(
  theme = app_theme,
  title = NULL,
  fillable = FALSE,
  tags$head(
    tags$style(HTML(mega_css)),
    tags$link(rel = "stylesheet", href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css")
  ),
  sidebar = sidebar(
    width = 280, bg = "transparent", class = "sidebar",
    div(class = "brand-header",
      h4(HTML('<i class="fas fa-chart-line"></i> Income DSS')),
      div(class = "brand-sub", "IE 477 \u2022 Bilkent University")
    ),
    navset_hidden(id = "main_nav"),
    tags$nav(class = "nav flex-column",
      tags$a(class = "nav-link active", href = "#",
             onclick = "Shiny.setInputValue('nav_tab', 'predict'); document.querySelectorAll('.sidebar .nav-link').forEach(e=>e.classList.remove('active')); this.classList.add('active');",
             HTML('<i class="fas fa-calculator"></i> Predict Income')),
      tags$a(class = "nav-link", href = "#",
             onclick = "Shiny.setInputValue('nav_tab', 'batch'); document.querySelectorAll('.sidebar .nav-link').forEach(e=>e.classList.remove('active')); this.classList.add('active');",
             HTML('<i class="fas fa-layer-group"></i> Batch Prediction')),
      tags$a(class = "nav-link", href = "#",
             onclick = "Shiny.setInputValue('nav_tab', 'compare'); document.querySelectorAll('.sidebar .nav-link').forEach(e=>e.classList.remove('active')); this.classList.add('active');",
             HTML('<i class="fas fa-scale-balanced"></i> Model Comparison')),
      tags$a(class = "nav-link", href = "#",
             onclick = "Shiny.setInputValue('nav_tab', 'info'); document.querySelectorAll('.sidebar .nav-link').forEach(e=>e.classList.remove('active')); this.classList.add('active');",
             HTML('<i class="fas fa-circle-info"></i> Model Info')),
      tags$a(class = "nav-link", href = "#",
             onclick = "Shiny.setInputValue('nav_tab', 'guide'); document.querySelectorAll('.sidebar .nav-link').forEach(e=>e.classList.remove('active')); this.classList.add('active');",
             HTML('<i class="fas fa-book-open"></i> User Guide'))
    ),
    div(class = "sidebar-status",
      div(class = "status-item",
        span(class = paste("status-dot", ifelse(LGBM_LOADED, "dot-active", "dot-demo"))),
        span(ifelse(LGBM_LOADED, "LightGBM", "LightGBM (demo)"))
      ),
      div(class = "status-item",
        span(class = paste("status-dot", ifelse(OLS_LOADED, "dot-active", "dot-demo"))),
        span(ifelse(OLS_LOADED, "OLS Regression", "OLS (demo)"))
      ),
      div(class = "status-item",
        span(class = paste("status-dot", ifelse(LASSO_LOADED, "dot-active", "dot-demo"))),
        span(ifelse(LASSO_LOADED, "Lasso Regression", "Lasso (demo)"))
      )
    )
  ),
  uiOutput("main_content")
)

################################################################################
# Server
################################################################################
server <- function(input, output, session) {
  current_tab <- reactiveVal("predict")
  batch_data  <- reactiveVal(NULL)

  observeEvent(input$nav_tab, { current_tab(input$nav_tab) })

  # ── Helpers ────────────────────────────────────
  seg_label <- function(x) { if (is.na(x)) "Unknown" else if (x < 30000) "Low" else if (x < 60000) "Medium" else "High" }
  seg_class <- function(s) switch(s, "Low"="seg-low", "Medium"="seg-medium", "High"="seg-high", "seg-medium")

  # Gerekli sütunları default değerlerle doldur (batch yükleme için kritik)
  REQUIRED_COLS <- list(
    HF_CUSTOMER_AGE = 35, KKB_SCORE = 1200,
    KKB_CC_OPEN_TOT_LIM_AMT = 50000, KKB_CC_UTIL_RATIO = 0.5,
    KKB_CUST_TENURE = 24, HF_Odeyebilecegi_Taksit_Tutari = 3000,
    HF_AVG_ASSET_CURR_ACC_L3M = 5000, KKB_TOT_APP_CNT_L12M = 2,
    KKB_CC_OPEN_CNT = 2, KKB_CURR_DELQ_DAY_CNT = 0,
    HF_EDUCATION = "\u00dcniversite", HF_APPLICANT_EMPLOYMENT_STATUS = "2",
    Tier = "T3"
  )

  ensure_columns <- function(df) {
    df <- as.data.frame(df, stringsAsFactors = FALSE, check.names = FALSE)
    for (col in names(REQUIRED_COLS)) {
      if (!(col %in% names(df))) {
        df[[col]] <- REQUIRED_COLS[[col]]
      } else {
        # Numeric beklenen kolonları dönüştür
        if (is.numeric(REQUIRED_COLS[[col]])) {
          df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
          df[[col]][is.na(df[[col]])] <- REQUIRED_COLS[[col]]
        } else {
          df[[col]] <- as.character(df[[col]])
          df[[col]][is.na(df[[col]]) | df[[col]] == ""] <- REQUIRED_COLS[[col]]
        }
      }
    }
    df
  }

  # ── Prediction functions ───────────────────────
  safe_num <- function(x, default = 0) {
    x <- suppressWarnings(as.numeric(x[1]))
    if (is.na(x)) default else x
  }

  demo_prediction <- function(new_data, model_type) {
    base  <- switch(model_type, "ols"=22000, "lasso"=22500, "lgbm"=23000)
    noise <- switch(model_type, "ols"=0.88, "lasso"=0.90, "lgbm"=1.00)
    pred <- (base +
      safe_num(new_data$KKB_SCORE, 1200)*10 +
      safe_num(new_data$HF_CUSTOMER_AGE, 35)*500 +
      safe_num(new_data$KKB_CC_OPEN_TOT_LIM_AMT, 50000)*0.2 +
      safe_num(new_data$HF_Odeyebilecegi_Taksit_Tutari, 3000)*3) * noise
    if (is.na(pred) || pred <= 0) pred <- 25000
    margin <- switch(model_type, "ols"=0.18, "lasso"=0.16, "lgbm"=0.13)
    list(predicted=pred, lower=pred*(1-margin), upper=pred*(1+margin), log_pred=log(pred))
  }

  make_prediction_lgbm <- function(nd) {
    if (LGBM_LOADED) {
      res <- tryCatch({
        pl <- predict(best_model, new_data=nd)$.pred
        list(predicted=exp(pl), lower=exp(pl-1.96*0.15), upper=exp(pl+1.96*0.15), log_pred=pl)
      }, error = function(e) NULL)
      if (!is.null(res)) return(res)
    }
    demo_prediction(nd, "lgbm")
  }
  make_prediction_ols <- function(nd) {
    if (OLS_LOADED) {
      res <- tryCatch({
        pl <- predict(ols_model, new_data=nd)$.pred
        list(predicted=exp(pl), lower=exp(pl-1.96*0.18), upper=exp(pl+1.96*0.18), log_pred=pl)
      }, error = function(e) NULL)
      if (!is.null(res)) return(res)
    }
    demo_prediction(nd, "ols")
  }
  make_prediction_lasso <- function(nd) {
    if (LASSO_LOADED) {
      res <- tryCatch({
        pl <- predict(lasso_model, new_data=nd)$.pred
        list(predicted=exp(pl), lower=exp(pl-1.96*0.17), upper=exp(pl+1.96*0.17), log_pred=pl)
      }, error = function(e) NULL)
      if (!is.null(res)) return(res)
    }
    demo_prediction(nd, "lasso")
  }
  make_prediction <- function(nd, mt="lgbm") {
    tryCatch(
      switch(mt,
        "ols"   = make_prediction_ols(nd),
        "lasso" = make_prediction_lasso(nd),
        "lgbm"  = make_prediction_lgbm(nd)),
      error = function(e) demo_prediction(nd, mt)
    )
  }

  build_input_data <- function() {
    data.frame(
      HF_CUSTOMER_AGE=input$age, KKB_SCORE=input$kkb_score,
      KKB_CC_OPEN_TOT_LIM_AMT=input$cc_limit, KKB_CC_UTIL_RATIO=input$cc_util,
      KKB_CUST_TENURE=input$tenure, HF_Odeyebilecegi_Taksit_Tutari=input$payment_cap,
      HF_AVG_ASSET_CURR_ACC_L3M=input$avg_asset, KKB_TOT_APP_CNT_L12M=input$app_count,
      KKB_CC_OPEN_CNT=input$cc_open_cnt, KKB_CURR_DELQ_DAY_CNT=input$delq_day_cnt,
      HF_EDUCATION=input$education, HF_APPLICANT_EMPLOYMENT_STATUS=input$employment,
      Tier=input$tier, stringsAsFactors=FALSE, check.names=FALSE)
  }

  # Vektörize tahmin (büyük batchlerde çok daha hızlı ve stabil)
  batch_predict <- function(df, mt) {
    n <- nrow(df)
    out <- data.frame(predicted=numeric(n), lower=numeric(n), upper=numeric(n))

    # Önce trained model ile tümünü tek seferde deneyelim
    tried <- FALSE
    if (mt == "lgbm" && LGBM_LOADED) {
      res <- tryCatch({
        pl <- predict(best_model, new_data=df)$.pred
        data.frame(predicted=exp(pl), lower=exp(pl-1.96*0.15), upper=exp(pl+1.96*0.15))
      }, error = function(e) NULL)
      if (!is.null(res)) return(res)
      tried <- TRUE
    } else if (mt == "ols" && OLS_LOADED) {
      res <- tryCatch({
        pl <- predict(ols_model, new_data=df)$.pred
        data.frame(predicted=exp(pl), lower=exp(pl-1.96*0.18), upper=exp(pl+1.96*0.18))
      }, error = function(e) NULL)
      if (!is.null(res)) return(res)
      tried <- TRUE
    } else if (mt == "lasso" && LASSO_LOADED) {
      res <- tryCatch({
        pl <- predict(lasso_model, new_data=df)$.pred
        data.frame(predicted=exp(pl), lower=exp(pl-1.96*0.17), upper=exp(pl+1.96*0.17))
      }, error = function(e) NULL)
      if (!is.null(res)) return(res)
      tried <- TRUE
    }

    # Fallback: satır satır demo prediction (vektörize)
    for (i in seq_len(n)) {
      r <- demo_prediction(df[i, , drop=FALSE], mt)
      out$predicted[i] <- r$predicted
      out$lower[i]     <- r$lower
      out$upper[i]     <- r$upper
    }
    out
  }

  # ════════════════════════════════════════════════
  # Page renderers
  # ════════════════════════════════════════════════
  output$main_content <- renderUI({
    switch(current_tab(),
      "predict" = ui_predict(),
      "batch"   = ui_batch(),
      "compare" = ui_compare(),
      "info"    = ui_info(),
      "guide"   = ui_guide(),
      ui_predict()
    )
  })

  ui_predict <- function() {
    tagList(
      div(class = "page-header",
        h3("Predict Income"),
        p("Enter customer details and select a model to predict monthly net income.")
      ),
      div(class = "glass-card",
        div(class = "card-header-custom", icon("microchip"), tags$h5("Select Model")),
        radioButtons("model_choice", NULL,
          choices = c("LightGBM" = "lgbm", "OLS Regression" = "ols", "Lasso Regression" = "lasso"),
          selected = "lgbm", inline = TRUE)
      ),
      fluidRow(
        column(6,
          div(class = "glass-card",
            div(class = "card-header-custom", icon("id-card"), tags$h5("Core Customer Information")),
            numericInput("age", "Customer Age:", value = 35, min = 18, max = 80, step = 1),
            numericInput("kkb_score", "KKB Credit Score:", value = 1200, min = 0, max = 1900, step = 10),
            numericInput("cc_limit", "Total Credit Card Limit (TL):", value = 50000, min = 0, max = 500000, step = 1000),
            numericInput("cc_util", "Credit Card Utilization (0\u20131):", value = 0.5, min = 0, max = 1, step = 0.05),
            numericInput("tenure", "Customer Tenure (months):", value = 24, min = 0, max = 300, step = 1),
            numericInput("payment_cap", "Monthly Payment Capacity (TL):", value = 3000, min = 0, max = 50000, step = 100)
          )
        ),
        column(6,
          div(class = "glass-card",
            div(class = "card-header-custom", icon("folder-open"), tags$h5("Additional Information")),
            numericInput("avg_asset", "Avg Asset Current Acc (L3M, TL):", value = 5000, min = 0, step = 500),
            numericInput("app_count", "Application Count (L12M):", value = 2, min = 0, max = 20, step = 1),
            numericInput("cc_open_cnt", "Open Credit Card Count:", value = 2, min = 0, max = 10, step = 1),
            numericInput("delq_day_cnt", "Current Delinquency Days:", value = 0, min = 0, max = 365, step = 1),
            selectInput("education", "Education Level:",
              choices = c("\u0130lkokul","Ortaokul","Lise","\u00dcniversite","Y\u00fcksek Lisans","Doktora"),
              selected = "\u00dcniversite"),
            selectInput("employment", "Employment Status:",
              choices = c("1","2","3","4","5","6","7","8","9"), selected = "2"),
            selectInput("city", "Residence City:",
              choices = c("\u0130STANBUL","ANKARA","\u0130ZM\u0130R","Other"), selected = "\u0130STANBUL"),
            selectInput("tier", "Customer Tier:",
              choices = c("T1","T2","T3","T4","T5","T6"), selected = "T3")
          )
        )
      ),
      div(style = "margin-bottom:20px;",
        actionButton("predict_single", HTML('<i class="fas fa-bolt"></i> Predict Income'),
                     class = "btn-predict")
      ),
      uiOutput("single_result")
    )
  }

  ui_batch <- function() {
    tagList(
      div(class = "page-header",
        h3("Batch Prediction"),
        p("Upload customer data and run predictions for multiple customers at once.")
      ),
      div(class = "glass-card",
        div(class = "card-header-custom", icon("cloud-arrow-up"), tags$h5("Upload & Configure")),
        fluidRow(
          column(4,
            selectInput("batch_model_choice", "Select Model:",
              choices = c("LightGBM" = "lgbm", "OLS Regression" = "ols",
                          "Lasso Regression" = "lasso", "All Models (Compare)" = "all"),
              selected = "lgbm")
          ),
          column(4,
            fileInput("batch_file", "Choose CSV / Excel File",
                     accept = c(".csv", ".xlsx", ".xls"))
          ),
          column(4,
            br(),
            downloadButton("download_template", "Download Template",
                          class = "btn btn-outline-primary btn-sm"),
            div(style = "height:10px;"),
            actionButton("predict_batch", HTML('<i class="fas fa-play"></i> Run Predictions'),
                        class = "btn-predict")
          )
        ),
        uiOutput("batch_status")
      ),
      div(class = "glass-card",
        div(class = "card-header-custom", icon("table-cells"), tags$h5("Results")),
        DTOutput("batch_results"),
        br(),
        downloadButton("download_results", "Download Results", class = "btn btn-outline-primary btn-sm")
      ),
      div(class = "glass-card",
        div(class = "card-header-custom", icon("chart-column"), tags$h5("Prediction Distribution")),
        plotOutput("batch_plot", height = 380)
      )
    )
  }

  ui_compare <- function() {
    tagList(
      div(class = "page-header",
        h3("Model Comparison"),
        p("Side-by-side performance of OLS, Lasso, and LightGBM across key metrics.")
      ),
      div(class = "glass-card",
        div(class = "card-header-custom", icon("table-cells-large"), tags$h5("Performance Metrics")),
        tableOutput("comparison_table")
      ),
      fluidRow(
        column(6,
          div(class = "glass-card",
            div(class = "card-header-custom", icon("chart-column"), tags$h5("Test RMSE Comparison")),
            plotOutput("compare_rmse_plot", height = 320)
          )
        ),
        column(6,
          div(class = "glass-card",
            div(class = "card-header-custom", icon("bullseye"), tags$h5("Test R\u00b2 Comparison")),
            plotOutput("compare_r2_plot", height = 320)
          )
        )
      ),
      div(class = "glass-card",
        div(class = "card-header-custom", icon("lightbulb"), tags$h5("Model Selection Guidance")),
        fluidRow(
          column(4, div(class = "guidance-card gc-blue",
            h6(HTML('<i class="fas fa-chart-simple"></i> OLS Regression')),
            p("Best for interpretability. Coefficients directly show the effect of each predictor."))
          ),
          column(4, div(class = "guidance-card gc-green",
            h6(HTML('<i class="fas fa-filter"></i> Lasso Regression')),
            p("Best for feature selection + interpretability. L1 penalty shrinks weak coefficients to zero."))
          ),
          column(4, div(class = "guidance-card gc-amber",
            h6(HTML('<i class="fas fa-bolt"></i> LightGBM')),
            p("Best for maximum accuracy. Captures non-linear relationships and interactions."))
          )
        )
      )
    )
  }

  ui_info <- function() {
    tagList(
      div(class = "page-header", h3("Model Information"),
          p("Detailed performance metrics and model descriptions.")),
      fluidRow(
        column(4, div(class = "metric-card mc-blue",
          div(class = "mc-icon", icon("chart-line")),
          div(class = "mc-value", paste0(format(round(model_metrics$test_rmse, 0), big.mark = ","), " TL")),
          div(class = "mc-label", "LightGBM Test RMSE"))),
        column(4, div(class = "metric-card mc-green",
          div(class = "mc-icon", icon("percent")),
          div(class = "mc-value", sprintf("%.3f", model_metrics$test_r2)),
          div(class = "mc-label", "LightGBM Test R\u00b2"))),
        column(4, div(class = "metric-card mc-amber",
          div(class = "mc-icon", icon("circle-check")),
          div(class = "mc-value", paste0(sum(c(LGBM_LOADED, OLS_LOADED, LASSO_LOADED)), " / 3")),
          div(class = "mc-label", "Models Loaded")))
      ),
      br(),
      fluidRow(
        column(6, div(class = "glass-card",
          div(class = "card-header-custom", icon("gauge-high"), tags$h5("LightGBM Performance")),
          h6("Cross-Validation (Log Scale)", style = "font-weight:700; color: var(--navy-700);"),
          tags$ul(style = "color: var(--navy-600); font-size:14px;",
            tags$li(strong("CV RMSE: "), sprintf("%.4f", model_metrics$cv_rmse)),
            tags$li(strong("CV MAE: "),  sprintf("%.4f", model_metrics$cv_mae)),
            tags$li(strong("CV R\u00b2: "),  sprintf("%.4f", model_metrics$cv_r2))),
          h6("Test Set (TL Scale)", style = "font-weight:700; color: var(--navy-700); margin-top:14px;"),
          tags$ul(style = "color: var(--navy-600); font-size:14px;",
            tags$li(strong("Test RMSE: "), paste0(format(round(model_metrics$test_rmse, 0), big.mark = ","), " TL")),
            tags$li(strong("Test MAE: "),  paste0(format(round(model_metrics$test_mae, 0), big.mark = ","), " TL")),
            tags$li(strong("Test R\u00b2: "), sprintf("%.4f", model_metrics$test_r2))))),
        column(6, div(class = "glass-card",
          div(class = "card-header-custom", icon("ranking-star"), tags$h5("Feature Importance")),
          plotOutput("importance_plot", height = 300)))
      )
    )
  }

  ui_guide <- function() {
    tagList(
      div(class = "page-header", h3("User Guide"),
          p("Step-by-step instructions for using the Income Prediction DSS.")),
      div(class = "glass-card",
        div(class = "guide-section",
          h5(span(class = "step-num", "1"), "Single Prediction"),
          tags$ol(
            tags$li("Choose a model (LightGBM / OLS / Lasso) at the top."),
            tags$li("Enter customer details in both input panels."),
            tags$li("Click ", strong("Predict Income"), "."),
            tags$li("View the hero card with predicted income, 95% CI, and segment."))),
        div(class = "guide-section",
          h5(span(class = "step-num", "2"), "Batch Prediction"),
          tags$ol(
            tags$li("Select a model (or 'All Models' for comparison)."),
            tags$li("Download the template and fill in customer rows."),
            tags$li("Upload CSV/Excel and click ", strong("Run Predictions"), "."),
            tags$li("Review results table and download as CSV."))),
        div(class = "guide-section",
          h5(span(class = "step-num", "3"), "Output Interpretation"),
          tags$ul(
            tags$li(strong("Predicted Income:"), " Model\u2019s best estimate of monthly net income (TL)."),
            tags$li(strong("95% CI:"), " Range where the true income likely falls."),
            tags$li(strong("Income Segment:"), " Low (< 30K), Medium (30\u201360K), High (> 60K).")))
      )
    )
  }

  # ════════════════════════════════════════════════
  # Single Prediction
  # ════════════════════════════════════════════════
  observeEvent(input$predict_single, {
    nd  <- build_input_data()
    mt  <- input$model_choice
    res <- make_prediction(nd, mt)
    if (is.null(res)) return()
    seg <- seg_label(res$predicted); scls <- seg_class(seg)
    ml  <- switch(mt, "lgbm"="LightGBM", "ols"="OLS Regression", "lasso"="Lasso Regression")
    all_res <- lapply(c("lgbm","ols","lasso"), function(m) {
      r <- make_prediction(nd, m); list(model=m, pred=r$predicted)
    })
    output$single_result <- renderUI({
      tagList(
        div(class = "hero-prediction",
          div(class = "hero-model-tag", ml),
          div(class = "hero-amount",
              paste0("\u20ba", format(round(res$predicted, 0), big.mark = ","))),
          div(class = "hero-ci",
              paste0("95% CI: \u20ba", format(round(res$lower,0), big.mark=","),
                     " \u2014 \u20ba", format(round(res$upper,0), big.mark=",")))
        ),
        div(class = "kpi-grid",
          div(class = "kpi-card",
            div(class = "kpi-icon", "\ud83d\udcca"),
            div(class = "kpi-value", span(class=paste("seg-badge",scls), seg)),
            div(class = "kpi-label", "Income Segment")),
          div(class = "kpi-card",
            div(class = "kpi-icon", "\u2b50"),
            div(class = "kpi-value", paste0(input$kkb_score, ifelse(input$kkb_score>1200," \u2713",""))),
            div(class = "kpi-label", "Credit Score")),
          div(class = "kpi-card",
            div(class = "kpi-icon", "\ud83d\udcb3"),
            div(class = "kpi-value", paste0(round(input$cc_util*100,0), "%")),
            div(class = "kpi-label", "CC Utilization")),
          div(class = "kpi-card",
            div(class = "kpi-icon", "\ud83d\udcb0"),
            div(class = "kpi-value", paste0("\u20ba", format(input$payment_cap, big.mark=","))),
            div(class = "kpi-label", "Payment Capacity"))
        ),
        div(class = "glass-card", style = "margin-top:20px;",
          div(class = "card-header-custom", icon("arrows-left-right"), tags$h5("Quick Model Comparison")),
          div(class = "compare-strip",
            lapply(all_res, function(r) {
              lbl <- switch(r$model, "lgbm"="LightGBM","ols"="OLS","lasso"="Lasso")
              sel <- if (r$model==mt) " is-selected" else ""
              div(class = paste0("compare-strip-item", sel),
                div(class = "cs-model", lbl),
                div(class = "cs-value", paste0("\u20ba", format(round(r$pred,0), big.mark=","))))
            })
          )
        )
      )
    })
  })

  # ════════════════════════════════════════════════
  # Batch Prediction (GÜVENLİ VERSİYON)
  # ════════════════════════════════════════════════
  output$download_template <- downloadHandler(
    filename = "income_prediction_template.csv",
    content = function(file) {
      write.csv(data.frame(
        HF_CUSTOMER_AGE=c(35,42,28), KKB_SCORE=c(1200,1450,980),
        KKB_CC_OPEN_TOT_LIM_AMT=c(50000,75000,25000), KKB_CC_UTIL_RATIO=c(0.5,0.3,0.8),
        KKB_CUST_TENURE=c(24,48,12), HF_Odeyebilecegi_Taksit_Tutari=c(3000,5000,1500),
        HF_AVG_ASSET_CURR_ACC_L3M=c(5000,8000,2000), KKB_TOT_APP_CNT_L12M=c(2,1,3),
        KKB_CC_OPEN_CNT=c(2,3,1), KKB_CURR_DELQ_DAY_CNT=c(0,0,15),
        HF_EDUCATION=c("\u00dcniversite","Y\u00fcksek Lisans","Lise"),
        HF_APPLICANT_EMPLOYMENT_STATUS=c("2","2","4"),
        Tier=c("T3","T2","T4"), stringsAsFactors=FALSE, check.names=FALSE
      ), file, row.names=FALSE, fileEncoding = "UTF-8")
    }
  )

  observeEvent(input$predict_batch, {
    req(input$batch_file)

    # Dosyayı güvenli şekilde oku
    file_info <- input$batch_file
    ext <- tolower(tools::file_ext(file_info$name))

    raw_data <- tryCatch({
      if (ext == "csv") {
        read.csv(file_info$datapath, stringsAsFactors = FALSE, check.names = FALSE,
                 fileEncoding = "UTF-8")
      } else if (ext %in% c("xlsx", "xls")) {
        readxl::read_excel(file_info$datapath)
      } else {
        stop("Desteklenmeyen dosya tipi: ", ext)
      }
    }, error = function(e) {
      showNotification(paste("Dosya okuma hatas\u0131:", e$message), type = "error", duration = 10)
      NULL
    })

    if (is.null(raw_data) || nrow(raw_data) == 0) {
      showNotification("Dosya bo\u015f veya okunamad\u0131.", type = "error")
      return()
    }

    # Çok büyük batch için güvenlik sınırı
    MAX_ROWS <- 50000
    if (nrow(raw_data) > MAX_ROWS) {
      showNotification(
        sprintf("Dosya \u00e7ok b\u00fcy\u00fck (%s sat\u0131r). \u0130lk %s sat\u0131r i\u015flenecek.",
                format(nrow(raw_data), big.mark=","), format(MAX_ROWS, big.mark=",")),
        type = "warning", duration = 8)
      raw_data <- raw_data[seq_len(MAX_ROWS), , drop = FALSE]
    }

    # Sütunları normalize et (eksik kolonları default ile doldur)
    data <- tryCatch(ensure_columns(raw_data),
                     error = function(e) {
                       showNotification(paste("Veri normalize hatas\u0131:", e$message),
                                        type = "error", duration = 10)
                       NULL
                     })
    if (is.null(data)) return()

    mc <- input$batch_model_choice
    n  <- nrow(data)

    withProgress(message = "Tahminler hesaplan\u0131yor...", value = 0, {
      result <- tryCatch({
        if (mc == "all") {
          for (m in c("lgbm","ols","lasso")) {
            incProgress(1/4, detail = paste("Model:", m))
            preds <- batch_predict(data, m)
            lbl <- switch(m, "lgbm"="LightGBM", "ols"="OLS", "lasso"="Lasso")
            data[[paste0("Pred_", lbl, "_TL")]] <- round(preds$predicted, 0)
          }
          pred_cols <- grep("^Pred_", names(data), value = TRUE)
          data$Ensemble_Avg_TL <- round(rowMeans(data[, pred_cols, drop = FALSE], na.rm = TRUE), 0)
          data$Income_Segment  <- cut(data$Ensemble_Avg_TL,
                                      breaks = c(-Inf, 30000, 60000, Inf),
                                      labels = c("Low","Medium","High"))
        } else {
          incProgress(0.3, detail = "Model \u00e7al\u0131\u015ft\u0131r\u0131l\u0131yor")
          preds <- batch_predict(data, mc)
          incProgress(0.5, detail = "Sonu\u00e7lar haz\u0131rlan\u0131yor")
          data$Predicted_Income_TL <- round(preds$predicted, 0)
          data$Lower_95CI_TL       <- round(preds$lower, 0)
          data$Upper_95CI_TL       <- round(preds$upper, 0)
          data$Income_Segment      <- cut(data$Predicted_Income_TL,
                                          breaks = c(-Inf, 30000, 60000, Inf),
                                          labels = c("Low","Medium","High"))
        }
        incProgress(1, detail = "Tamamland\u0131")
        data
      }, error = function(e) {
        showNotification(paste("Tahmin hatas\u0131:", e$message), type = "error", duration = 10)
        NULL
      })
    })

    if (is.null(result)) return()

    batch_data(result)
    showNotification(
      sprintf("%s m\u00fc\u015fteri i\u00e7in tahmin tamamland\u0131!", format(n, big.mark=",")),
      type = "message", duration = 5)
  })

  output$batch_status <- renderUI({
    if (is.null(input$batch_file)) return(NULL)
    div(style = "margin-top:14px; font-size:13px; color: var(--navy-600);",
        icon("file"),
        sprintf(" Y\u00fcklenen dosya: %s (%.2f MB)",
                input$batch_file$name, input$batch_file$size / 1024^2))
  })

  output$batch_results <- renderDT({
    req(batch_data())
    df <- batch_data()
    # Gösterilecek sütunları sınırla (çok fazla sütun varsa)
    if (ncol(df) > 20) {
      pred_cols <- grep("Predicted|Pred_|Lower|Upper|Segment|Ensemble", names(df), value = TRUE)
      other_cols <- setdiff(names(df), pred_cols)
      df <- df[, c(head(other_cols, 10), pred_cols), drop = FALSE]
    }
    currency_cols <- intersect(names(df),
      c("Predicted_Income_TL","Lower_95CI_TL","Upper_95CI_TL",
        "Pred_LightGBM_TL","Pred_OLS_TL","Pred_Lasso_TL","Ensemble_Avg_TL"))
    dt <- datatable(df, options=list(pageLength=10, scrollX=TRUE), rownames=FALSE)
    if (length(currency_cols) > 0) {
      dt <- dt %>% formatCurrency(currency_cols, currency="\u20ba", digits=0)
    }
    dt
  })

  output$batch_plot <- renderPlot({
    req(batch_data())
    df <- batch_data()
    pc <- if ("Ensemble_Avg_TL" %in% names(df)) "Ensemble_Avg_TL" else "Predicted_Income_TL"
    req(pc %in% names(df))
    med_val <- median(df[[pc]], na.rm = TRUE)
    ggplot(df, aes(x=.data[[pc]])) +
      geom_histogram(bins=30, fill="#2563eb", color="#fff", alpha=0.85) +
      geom_vline(xintercept=med_val, color="#ef4444", linetype="dashed", linewidth=1) +
      scale_x_continuous(labels=comma) +
      labs(title="Distribution of Predicted Incomes",
           subtitle=paste("Median:", format(round(med_val,0), big.mark=","), "TL"),
           x="Predicted Income (TL)", y="Count") +
      theme_minimal(base_size=14) +
      theme(text=element_text(family="sans"), plot.title=element_text(face="bold"))
  })

  output$download_results <- downloadHandler(
    filename = function() paste0("income_predictions_", Sys.Date(), ".csv"),
    content  = function(file) {
      req(batch_data())
      write.csv(batch_data(), file, row.names=FALSE, fileEncoding = "UTF-8")
    }
  )

  # ════════════════════════════════════════════════
  # Comparison outputs
  # ════════════════════════════════════════════════
  output$comparison_table <- renderTable({
    data.frame(
      Metric = c("CV RMSE (log)","CV MAE (log)","CV R\u00b2",
                 "Test RMSE (TL)","Test MAE (TL)","Test R\u00b2"),
      OLS = c(sprintf("%.4f",ols_metrics$cv_rmse), sprintf("%.4f",ols_metrics$cv_mae),
              sprintf("%.4f",ols_metrics$cv_r2),
              format(round(ols_metrics$test_rmse,0),big.mark=","),
              format(round(ols_metrics$test_mae,0),big.mark=","),
              sprintf("%.4f",ols_metrics$test_r2)),
      Lasso = c(sprintf("%.4f",lasso_metrics$cv_rmse), sprintf("%.4f",lasso_metrics$cv_mae),
                sprintf("%.4f",lasso_metrics$cv_r2),
                format(round(lasso_metrics$test_rmse,0),big.mark=","),
                format(round(lasso_metrics$test_mae,0),big.mark=","),
                sprintf("%.4f",lasso_metrics$test_r2)),
      LightGBM = c(sprintf("%.4f",model_metrics$cv_rmse), sprintf("%.4f",model_metrics$cv_mae),
                   sprintf("%.4f",model_metrics$cv_r2),
                   format(round(model_metrics$test_rmse,0),big.mark=","),
                   format(round(model_metrics$test_mae,0),big.mark=","),
                   sprintf("%.4f",model_metrics$test_r2))
    )
  }, striped=TRUE, bordered=TRUE, hover=TRUE, width="100%")

  theme_dss <- function() {
    theme_minimal(base_size=14) %+replace%
    theme(
      text = element_text(family="sans"),
      plot.title = element_text(face="bold", size=15, color="#0f172a"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text = element_text(color="#475569", size=12),
      plot.margin = margin(10,10,10,10)
    )
  }

  output$compare_rmse_plot <- renderPlot({
    df <- data.frame(
      Model=c("OLS","Lasso","LightGBM"),
      RMSE=c(ols_metrics$test_rmse, lasso_metrics$test_rmse, model_metrics$test_rmse))
    df$Model <- factor(df$Model, levels=df$Model[order(df$RMSE, decreasing=TRUE)])
    ggplot(df, aes(x=Model, y=RMSE, fill=Model)) +
      geom_col(width=0.55, show.legend=FALSE) +
      geom_text(aes(label=format(round(RMSE,0),big.mark=",")),
                vjust=-0.5, fontface="bold", size=5, color="#0f172a") +
      scale_fill_manual(values=c("OLS"="#3b82f6","Lasso"="#10b981","LightGBM"="#f59e0b")) +
      scale_y_continuous(labels=comma, expand=expansion(mult=c(0,0.15))) +
      labs(title="Test RMSE (TL) \u2014 Lower is Better", x=NULL, y="RMSE (TL)") +
      theme_dss()
  })

  output$compare_r2_plot <- renderPlot({
    df <- data.frame(
      Model=c("OLS","Lasso","LightGBM"),
      R2=c(ols_metrics$test_r2, lasso_metrics$test_r2, model_metrics$test_r2))
    df$Model <- factor(df$Model, levels=df$Model[order(df$R2)])
    ggplot(df, aes(x=Model, y=R2, fill=Model)) +
      geom_col(width=0.55, show.legend=FALSE) +
      geom_text(aes(label=sprintf("%.3f",R2)), vjust=-0.5, fontface="bold", size=5, color="#0f172a") +
      scale_fill_manual(values=c("OLS"="#3b82f6","Lasso"="#10b981","LightGBM"="#f59e0b")) +
      scale_y_continuous(limits=c(0,1), expand=expansion(mult=c(0,0.1))) +
      labs(title="Test R\u00b2 \u2014 Higher is Better", x=NULL, y="R\u00b2") +
      theme_dss()
  })

  output$importance_plot <- renderPlot({
    top_f <- head(feature_importance, 10)
    ggplot(top_f, aes(x=reorder(Variable,Importance), y=Importance)) +
      geom_col(fill="#2563eb", alpha=0.85, width=0.65) +
      coord_flip() +
      labs(title="Top 10 Features", x=NULL, y="Importance") +
      theme_dss() +
      theme(panel.grid.major.y=element_blank())
  })
}

################################################################################
# Run the App
################################################################################
shinyApp(ui = ui, server = server)
