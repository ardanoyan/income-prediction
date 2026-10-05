#!/usr/bin/env python3
"""
make_synthetic_data.py: write a synthetic customer table with the schema of the original
Hayat Finans extract, so that the analysis notebook and the Shiny apps can be run end to end
without the real data.

Nothing here is derived from real customers. Every column is drawn from simple parametric
distributions, and the target (Edevlet_Net_Gelir, monthly net income in TL) is generated from a
planted relationship with age, bureau score, credit-card limit, education, employment status
and city, plus noise. The numbers the notebook produces on this data are therefore illustrative
only; the results of the actual project are summarised in README.md and docs/HYFGT_poster.pdf.

Usage:
    python scripts/make_synthetic_data.py            # 5000 rows, seed 477
    python scripts/make_synthetic_data.py 20000 1    # rows, seed

Writes:
    data/synthetic_customers.xlsx                   read by income-prediction.qmd
    data/il_ortalama_kira.xlsx                      synthetic province rent table (sheet kira_verileri_canli_kayit)
    shiny/extreme_values/data/synthetic_customers.xlsx   a copy for the Shiny app
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 477
rng = np.random.default_rng(SEED)

MISSING = -999999999999.99  # sentinel the original extract used for "not available"

PROVINCES = [
    "ADANA", "ADIYAMAN", "AFYONKARAHİSAR", "AĞRI", "AMASYA", "ANKARA", "ANTALYA", "ARTVİN", "AYDIN",
    "BALIKESİR", "BİLECİK", "BİNGÖL", "BİTLİS", "BOLU", "BURDUR", "BURSA", "ÇANAKKALE", "ÇANKIRI",
    "ÇORUM", "DENİZLİ", "DİYARBAKIR", "EDİRNE", "ELAZIĞ", "ERZİNCAN", "ERZURUM", "ESKİŞEHİR",
    "GAZİANTEP", "GİRESUN", "GÜMÜŞHANE", "HAKKARİ", "HATAY", "ISPARTA", "MERSİN", "İSTANBUL", "İZMİR",
    "KARS", "KASTAMONU", "KAYSERİ", "KIRKLARELİ", "KIRŞEHİR", "KOCAELİ", "KONYA", "KÜTAHYA", "MALATYA",
    "MANİSA", "KAHRAMANMARAŞ", "MARDİN", "MUĞLA", "MUŞ", "NEVŞEHİR", "NİĞDE", "ORDU", "RİZE", "SAKARYA",
    "SAMSUN", "SİİRT", "SİNOP", "SİVAS", "TEKİRDAĞ", "TOKAT", "TRABZON", "TUNCELİ", "ŞANLIURFA", "UŞAK",
    "VAN", "YOZGAT", "ZONGULDAK", "AKSARAY", "BAYBURT", "KARAMAN", "KIRIKKALE", "BATMAN", "ŞIRNAK",
    "BARTIN", "ARDAHAN", "IĞDIR", "YALOVA", "KARABÜK", "KİLİS", "OSMANİYE", "DÜZCE",
]
BIG_CITY_WEIGHT = {"İSTANBUL": 18, "ANKARA": 7, "İZMİR": 5, "BURSA": 3, "ANTALYA": 3, "KOCAELİ": 2, "ADANA": 2}


def lognormal(median, sigma, size):
    return rng.lognormal(np.log(median), sigma, size)


def zero_inflated(p_zero, median, sigma, size, round_to=1):
    x = lognormal(median, sigma, size)
    x[rng.random(size) < p_zero] = 0.0
    return np.round(x / round_to) * round_to


def main():
    n = N
    # --- demographics -------------------------------------------------------------------------
    age = np.clip(rng.gamma(4.0, 4.5, n) + 18, 18, 82).round().astype(int)
    education = rng.choice(np.arange(1, 9), n, p=[0.05, 0.10, 0.12, 0.30, 0.08, 0.25, 0.07, 0.03])
    employment = rng.choice([1, 2, 3, 4, 5, 6, 8, 10], n, p=[0.30, 0.40, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03])
    profession = rng.integers(2, 108, n)
    work_title = rng.integers(1, 38, n)
    marital = rng.choice([0, 1, 2, 4], n, p=[0.45, 0.40, 0.10, 0.05])
    weights = np.array([BIG_CITY_WEIGHT.get(p, 1.0) for p in PROVINCES], dtype=float)
    city = rng.choice(PROVINCES, n, p=weights / weights.sum())
    city_idx = np.array([PROVINCES.index(c) for c in city])

    # --- credit bureau (KKB) block ------------------------------------------------------------
    score = np.clip(rng.normal(1050, 280, n), 0, 1900).round().astype(int)
    tenure = np.clip(rng.gamma(2.2, 60, n), 0, 425).round().astype(int)
    cc_open_cnt = rng.poisson(2.2, n)
    cc_limit = zero_inflated(0.12, 58000, 1.1, n, 100)
    cc_limit[cc_open_cnt == 0] = 0
    cc_max_limit = np.where(cc_limit > 0, np.round(cc_limit * rng.uniform(0.35, 1.0, n) / 100) * 100, 0)
    cc_util = np.where(cc_limit > 0, np.clip(rng.beta(1.6, 2.0, n), 0, 1), 0)
    cc_risk = np.round(cc_limit * cc_util)
    cc_monthly_paym = np.round(cc_risk * rng.uniform(0.05, 0.35, n))
    cc_acct_cnt = cc_open_cnt + rng.poisson(1.0, n)
    delq_acct = rng.poisson(2.5, n)
    non_delq = rng.poisson(9, n)
    cl_limit = zero_inflated(0.55, 90000, 1.0, n, 100)
    cl_inst = np.round(cl_limit * rng.uniform(0.02, 0.06, n))
    cl_risk = np.round(cl_limit * rng.uniform(0.2, 0.95, n))
    mortg_cnt = rng.choice([0, 1, 2], n, p=[0.88, 0.10, 0.02])
    mortg_limit = np.where(mortg_cnt > 0, np.round(lognormal(900000, 0.5, n) / 1000) * 1000, 0)
    mortg_risk = np.round(mortg_limit * rng.uniform(0.3, 0.95, n))
    od_cnt = rng.poisson(1.1, n)
    od_limit = np.where(od_cnt > 0, np.round(lognormal(6000, 1.0, n) / 100) * 100, 0)
    od_util = np.where(od_limit > 0, np.clip(rng.beta(1.2, 3.0, n), 0, 1), 0)
    od_risk = np.round(od_limit * od_util)
    app_cnt_12m = rng.poisson(3.0, n)
    delq_day = rng.choice([-9, 0, 1, 2, 3, 5, 7, 9], n, p=[0.05, 0.70, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02])
    curr_delq_amt = zero_inflated(0.78, 4000, 1.3, n)
    follow_amt = zero_inflated(0.90, 20000, 1.2, n)
    assets_curr = zero_inflated(0.60, 12000, 1.4, n)
    assets_part = zero_inflated(0.90, 30000, 1.3, n)
    assets_gold = zero_inflated(0.97, 40000, 1.0, n)

    # --- province rent table and the planted income model ---------------------------------------
    rent_median = 9000 + 15000 * (weights / weights.max()) ** 0.5  # bigger cities, higher rent
    rent_city = rent_median * rng.lognormal(0, 0.08, len(PROVINCES))
    edu_effect = np.array([0, 0.00, 0.05, 0.10, 0.18, 0.22, 0.40, 0.55, 0.75])[education]
    emp_effect = {1: 0.00, 2: 0.08, 3: 0.20, 4: -0.10, 5: -0.05, 6: 0.15, 8: -0.20, 10: -0.30}
    emp_vec = np.array([emp_effect[e] for e in employment])
    prof_vec = rng.normal(0, 0.15, 110)[profession]
    log_income = (
        9.55
        + 0.030 * (age - 35) - 0.00045 * (age - 35) ** 2
        + 0.00055 * (score - 1050)
        + 0.16 * np.log1p(cc_limit / 1000.0)
        + edu_effect + emp_vec + prof_vec
        + 0.25 * np.log(rent_city[city_idx] / rent_city.mean())
        + rng.normal(0, 0.42, n)
    )
    income = np.clip(np.exp(log_income), 8000, 600000).round(2)
    observed = rng.random(n) < 0.6  # e-Devlet income is available for part of the applicants only
    net_income = np.where(observed, income, MISSING)
    gross_income = np.where(observed, np.round(income * 1.27, 2), MISSING)
    declared = np.round(income * rng.lognormal(0.1, 0.5, n) / 100) * 100
    model_income = np.round(income * rng.lognormal(-0.05, 0.35, n), 2)

    # --- application outcome block --------------------------------------------------------------
    talep = rng.choice([8933, 50000, 250000, 1000000], n, p=[0.02, 0.15, 0.13, 0.70])
    final_limit = np.where(rng.random(n) < 0.3, rng.choice([0, 10000, 25000, 50000, 100000], n), 0)
    final_gelir = np.where(final_limit > 0, income.round(0), 0)
    taksit = np.where(final_limit > 0, np.round(income * rng.uniform(0.25, 0.45, n) / 10) * 10, 0)
    tier = rng.choice([f"T{i}" for i in range(1, 9)], n, p=[0.03, 0.07, 0.12, 0.16, 0.20, 0.25, 0.12, 0.05])
    karar = rng.choice(["BELGE", "DL", "AC"], n, p=[0.55, 0.30, 0.15])
    reason_codes = ["KO01", "KO02", "KO03", "KO04", "REJ01", "REJ07", "REJ08", "RE05", "EUB04", "EUB05",
                    "EUB28", "EUB29", "EUB38", "FR09"]
    app_dates = pd.to_datetime("2025-06-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
    period = app_dates.strftime("%Y/%m")
    emp_dates = (app_dates - pd.to_timedelta(rng.integers(30, 9000, n), unit="D")).strftime("%Y%m%d").astype(int)

    df = pd.DataFrame({
        "AYRAC1": "BASVURU_BILGILERI>>>>",
        "Versiyon_Bireysel": 20250615,
        "PowerCurveCallId": np.arange(4_200_000, 4_200_000 + n),
        "SystemDate": (app_dates + pd.to_timedelta(rng.integers(0, 86400, n), unit="s")).strftime("%Y-%m-%dT%H:%M:%S.000000"),
        "HF_APPLICATION_DATE": app_dates.strftime("%Y%m%d").astype(int),
        "Sıra": np.arange(1, n + 1),
        "Final_Karar": karar,
        "HF_POWERCURVE_CALL_REASON_CODE": 4,
        "HF_CHANNEL_TYPE": 5,
        "HF_CREDIT_TYPE": 150,
        "AYRAC2": "MUSTERI_BILGILERI>>>>",
        "SEGMENT": 2,
        "HF_WHITELIST_TYPE": -9,
        "HF_STAFF_FL": 0,
        "HF_APS_CITY_CODE": city_idx + 1,
        "İkamet İl": city,
        "HF_APPLICANT_EMPLOYMENT_STATUS": employment,
        "HF_APPLICANT_PROFESSION_CODE": profession,
        "HF_CUST_WORK_TITLE": work_title,
        "HF_CUSTOMER_AGE": age,
        "HF_EDUCATION": education,
        "VARLIKLI_FLG": (assets_curr + assets_part + assets_gold > 500000).astype(int),
        "HF_ASSET_PRTCPATION_ACC_L3M": assets_part,
        "HF_AVG_ASSET_CURR_ACC_L3M": assets_curr,
        "HF_AVG_ASSET_GOLD_ACC_L3M": assets_gold,
        "AYRAC3": "KKB_VERILERI>>>>",
        "KKB_SCORE": score,
        "KKB_CUST_TENURE": tenure,
        "HF_DATE_OF_EMPLOYMENT": np.where(rng.random(n) < 0.35, 19000101, emp_dates),
        "KKB_ALL_CC_DELQ_CNT_L12M": rng.poisson(2.5, n),
        "KKB_ALL_CL_DELQ_CNT_L12M": rng.poisson(0.8, n),
        "KKB_ALL_DELQ_AMT_CURR": curr_delq_amt,
        "KKB_CURR_GRT30D_DELQ_TOT_AMT": zero_inflated(0.9, 5000, 1.2, n),
        "KKB_ALL_DELQ_T1_CNT_L3M": rng.poisson(0.5, n),
        "KKB_ALL_DELQ_T2_CNT_L12M": rng.poisson(0.4, n),
        "KKB_ALL_PROD_WST_STA_PERF_EVER": rng.choice(["L", "U", "1", "2", "3", "4", "5"], n, p=[0.45, 0.20, 0.15, 0.08, 0.05, 0.04, 0.03]),
        "KKB_CA_3_6_NO_OF_ACCT_L12M": rng.poisson(0.6, n),
        "KKB_CC_DELQ_ACCT_CNT": rng.poisson(0.4, n),
        "KKB_CC_OPEN_CNT": cc_open_cnt,
        "KKB_CC_ACCT_CNT": cc_acct_cnt,
        "KKB_CL_DELQ_T1PLUS_CNT4_L6M": rng.poisson(0.3, n),
        "KKB_CURR_CC_DELQ_AMT": zero_inflated(0.82, 3000, 1.3, n),
        "KKB_CURR_DELQ_DAY_CNT": delq_day,
        "KKB_CURR_TOT_DELQ_AMT": curr_delq_amt + zero_inflated(0.9, 2000, 1.0, n),
        "KKB_DELQ_ACCNT_CNT_L12M": rng.poisson(2.0, n),
        "KKB_DELQ_ACCT_CNT": delq_acct,
        "KKB_DELQ_DAY_CURR_WST_AMT": zero_inflated(0.85, 4000, 1.3, n),
        "KKB_INST_OPEN_ALL_PROD_RISK": cl_inst,
        "KKB_MORTG_TOT_LIM_AMT": mortg_limit,
        "KKB_MORTG_TOT_RISK_AMT": mortg_risk,
        "KKB_NON_DELQ_ACCT_CNT": non_delq,
        "KKB_OD_OPEN_DELQ_ACCT_CNT": rng.poisson(0.2, n),
        "KKB_OPEN_CL_WST_L6M": rng.choice([0, 1, 2, 3, 9], n, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
        "KKB_OPEN_GPL_ACCT_CNT": rng.poisson(0.8, n),
        "KKB_OPEN_MORTG_CNT": mortg_cnt,
        "KKB_OPEN_OD_CNT": od_cnt,
        "KKB_WST_STA_EVER_PERF_L3M": rng.choice([0, 1, 2, 3, 9], n, p=[0.4, 0.4, 0.1, 0.05, 0.05]),
        "HF_APPLICANT_MARITAL_STATU": marital,
        "HF_MATURITY": rng.choice([6, 12], n, p=[0.2, 0.8]),
        "KKB_OD_TOT_RISK_AMT": od_risk,
        "KKB_OD_OPEN_TOT_LIM": od_limit,
        "KKB_CC_OPEN_TOT_LIM_AMT": cc_limit,
        "KKB_CC_OPEN_TOT_RISK": cc_risk,
        "KKB_CC_TOT_MONTHLY_PAYM_AMT": cc_monthly_paym,
        "KKB_REJECTED_APP_L3M": 0,
        "KKB_TOT_APP_CNT_L12M": app_cnt_12m,
        "KKB_L12M_FOLLOW_AMT": follow_amt,
        "KKB_CURR_TOT_FOLLOW_AMT": follow_amt * rng.uniform(0.8, 1.0, n),
        "AYRAC5": "RISK_BILGILERI>>>>",
        "KKB_KAPS_USED_CRD_INST_AMT": 0,
        "HF_APPROVED_AL_MONTHLY_INST_AMT": MISSING,
        "HF_APPROVED_GPL_MONTHLY_INST_AMT": np.where(rng.random(n) < 0.02, 12210.0, MISSING),
        "HF_APPROVED_MORTG_MNTLY_INST_AMT": MISSING,
        "HF_TOT_AL_MONTHLY_INST_AMT": MISSING,
        "HF_TOT_GPL_MONTHLY_INST_AMT": np.where(rng.random(n) < 0.05, np.round(lognormal(15000, 0.8, n)), MISSING),
        "HF_TOT_MORTG_MONTHLY_INST_AMT": MISSING,
        "HF_OPEN_CC_RISK": MISSING,
        "HF_CC_TOT_LIM": MISSING,
        "HF_SP_L12M_FOLLOW_AMT": MISSING,
        "KKB_CC_UTIL_RATIO": np.round(cc_util, 4),
        "KKB_CC_OPEN_MAX_LIM_AMT": cc_max_limit,
        "KKB_CL_OPEN_TOT_LIM_AMT": cl_limit,
        "KKB_CL_OPEN_TOT_INST_AMT": cl_inst,
        "KKB_CL_TOT_RISK_AMT": cl_risk,
        "KKB_OD_UTIL_RATIO": np.round(od_util, 4),
        "AYRAC6": "SM_CIKTILARI>>>>",
        "Talep_Limit": talep,
        "Final_Limit": final_limit,
        "Rating_Basvuru_Notu": 0,
        "Limit_Gecerlilik_Tarihi": (app_dates + pd.Timedelta(days=90)).strftime("%Y%m%d").astype(int),
        "Edevlet_Brut_Gelir": gross_income,
        "Edevlet_Net_Gelir": net_income,
        "HF_PERIOD": np.where(observed, period, "-999999"),
        "HF_QUERY_DATE": np.where(observed, app_dates.strftime("%Y%m%d").astype(int), 19000101),
        "Final_Gelir": final_gelir,
        "Beyan_Gelir": declared,
        "Model_Gelir": model_income,
        "Basvuru_Skoru": np.clip(rng.normal(140, 40, n), 39, 250).round().astype(int),
        "Ihtiyac_modeli_Final_skor": np.clip(rng.normal(660, 120, n), 367, 1000).round().astype(int),
        "HF_Odeyebilecegi_Taksit_Tutari": taksit,
        "VADE": rng.choice([6, 12], n, p=[0.2, 0.8]),
        "HF_APP_INST_AMT": rng.choice([581, 5810, 23240, 116200], n, p=[0.02, 0.08, 0.15, 0.75]),
    })
    for k in range(1, 10):
        col = rng.choice(reason_codes, n)
        col = np.where(rng.random(n) < min(0.02 + 0.09 * (k - 1), 0.9), None, col)
        df[f"Karar_Gerekcesi_{k}"] = col
    df["KKB_Risk_LM"] = zero_inflated(0.8, 150000, 1.0, n)
    df["KKB_Risk_L12M"] = zero_inflated(0.5, 120000, 1.1, n)
    df["KKB_Limit_LM"] = zero_inflated(0.8, 150000, 1.0, n)
    df["KKB_WPS_L24M"] = rng.choice([0, 1, 2, 3, 5, 9], n, p=[0.3, 0.45, 0.1, 0.07, 0.05, 0.03])
    df["Edevlet_Skoru"] = np.where(observed, np.clip(rng.normal(60, 80, n), -40, 355).round(), 0).astype(int)
    df["PMML"] = np.round(rng.beta(1.2, 6, n), 4)
    df["CALLRep"] = 4
    df["CallIdRep"] = df["PowerCurveCallId"]
    df["Kullandırım_FL"] = "H"
    df["Kullandırım_Tutarı"] = np.nan
    df["Kullandırım_Tarihi"] = np.nan
    df["Tier"] = tier

    rent = pd.DataFrame({
        "Column1": np.repeat(PROVINCES, 3),
        "Column2": [f"İlçe {j + 1}" for _ in PROVINCES for j in range(3)],
        "Column3": [f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    for v in np.repeat(rent_city, 3) * rng.lognormal(0, 0.12, 3 * len(PROVINCES))],
    })

    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    df.to_excel(out / "synthetic_customers.xlsx", index=False, sheet_name="Sheet3")
    with pd.ExcelWriter(out / "il_ortalama_kira.xlsx") as xw:
        rent.to_excel(xw, index=False, sheet_name="kira_verileri_canli_kayit")
    app_dir = ROOT / "shiny" / "extreme_values" / "data"
    app_dir.mkdir(parents=True, exist_ok=True)
    df.to_excel(app_dir / "synthetic_customers.xlsx", index=False, sheet_name="Sheet3")
    obs = df.loc[df["Edevlet_Net_Gelir"] > 0, "Edevlet_Net_Gelir"]
    print(f"wrote {len(df)} rows x {df.shape[1]} columns; income observed for {len(obs)} rows, "
          f"median {obs.median():,.0f} TL, p90 {obs.quantile(0.9):,.0f} TL")


if __name__ == "__main__":
    main()
