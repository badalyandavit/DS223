import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tools.numdiff import approx_hess


@dataclass
class AFTResult:
    distribution: str
    params: np.ndarray
    param_names: list
    log_likelihood: float
    aic: float
    bic: float
    success: bool
    message: str
    n: int


def prepare_data(df: pd.DataFrame):
    data = df.copy()
    data["event"] = data["churn"].astype(str).str.lower().eq("yes").astype(int)
    data["duration"] = data["tenure"].astype(float)

    numeric_cols = ["age", "address", "income"]
    cat_cols = ["region", "marital", "ed", "retire", "gender", "voice", "internet", "forward", "custcat"]

    scaler = {}
    for col in numeric_cols:
        mean = data[col].mean()
        std = data[col].std(ddof=0)
        scaler[col] = {"mean": float(mean), "std": float(std)}
        data[f"{col}_z"] = (data[col] - mean) / std

    x = pd.get_dummies(data[[f"{c}_z" for c in numeric_cols] + cat_cols], drop_first=True, dtype=float)
    x = x.loc[:, x.std(ddof=0) > 1e-12]

    return data, x, scaler


def neg_log_likelihood(params, distribution, xmat, t, e):
    beta = params[:xmat.shape[1]]
    eta = xmat @ beta

    if distribution == "exponential":
        scale = np.exp(eta)
        z = t / scale
        log_surv = -z
        log_pdf = -eta - z
    else:
        log_sigma = params[-1]
        sigma = np.exp(log_sigma)

        if distribution == "weibull":
            shape = 1.0 / sigma
            log_z = shape * (np.log(t) - eta)
            z = np.exp(np.clip(log_z, -700, 700))
            log_surv = -z
            log_pdf = np.log(shape) - eta + (shape - 1.0) * (np.log(t) - eta) - z

        elif distribution == "lognormal":
            z = (np.log(t) - eta) / sigma
            log_pdf = stats.norm.logpdf(z) - np.log(t) - log_sigma
            log_surv = stats.norm.logsf(z)

        elif distribution == "loglogistic":
            shape = 1.0 / sigma
            log_z = shape * (np.log(t) - eta)
            log_surv = -np.logaddexp(0, log_z)
            log_pdf = np.log(shape) - eta + (shape - 1.0) * (np.log(t) - eta) - 2.0 * np.logaddexp(0, log_z)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    ll = np.sum(e * log_pdf + (1 - e) * log_surv)
    if not np.isfinite(ll):
        return 1e100
    return -ll


def fit_aft(distribution, xmat, t, e, names):
    beta_start = np.linalg.lstsq(xmat, np.log(t), rcond=None)[0]
    p0 = beta_start if distribution == "exponential" else np.r_[beta_start, 0.0]

    result = optimize.minimize(
        neg_log_likelihood,
        p0,
        args=(distribution, xmat, t, e),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-6},
    )

    if not result.success:
        retry = optimize.minimize(
            neg_log_likelihood,
            result.x,
            args=(distribution, xmat, t, e),
            method="L-BFGS-B",
            options={"maxiter": 5000, "ftol": 1e-11, "gtol": 1e-7},
        )
        if retry.fun < result.fun:
            result = retry

    ll = -neg_log_likelihood(result.x, distribution, xmat, t, e)
    k = len(result.x)
    n = len(t)

    param_names = names.copy()
    if distribution != "exponential":
        param_names.append("log_sigma")

    return AFTResult(
        distribution=distribution,
        params=result.x,
        param_names=param_names,
        log_likelihood=float(ll),
        aic=float(2 * k - 2 * ll),
        bic=float(np.log(n) * k - 2 * ll),
        success=bool(result.success),
        message=str(result.message),
        n=n,
    )


def coefficient_table(model: AFTResult, distribution, xmat, t, e):
    hessian = approx_hess(model.params, lambda p: neg_log_likelihood(p, distribution, xmat, t, e))
    cov = np.linalg.inv(hessian)
    se = np.sqrt(np.diag(cov))
    z = model.params / se
    p = 2 * stats.norm.sf(np.abs(z))

    table = pd.DataFrame(
        {
            "term": model.param_names,
            "coef": model.params,
            "se": se,
            "z": z,
            "p_value": p,
            "time_ratio": np.exp(model.params),
        }
    )
    return table


def survival_matrix(distribution, params, xmat, months):
    beta = params[:xmat.shape[1]]
    eta = xmat @ beta
    tgrid = np.asarray(months, dtype=float)

    if distribution == "exponential":
        s = np.exp(-tgrid[:, None] / np.exp(eta)[None, :])
    else:
        sigma = np.exp(params[-1])
        if distribution == "weibull":
            shape = 1.0 / sigma
            s = np.exp(-((tgrid[:, None] / np.exp(eta)[None, :]) ** shape))
        elif distribution == "lognormal":
            z = (np.log(tgrid[:, None]) - eta[None, :]) / sigma
            s = stats.norm.sf(z)
        elif distribution == "loglogistic":
            shape = 1.0 / sigma
            s = 1.0 / (1.0 + (tgrid[:, None] / np.exp(eta)[None, :]) ** shape)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    return s.T


def plot_model_curves(models, x_reference, months, output_path):
    plt.figure(figsize=(9, 6))
    for distribution, model in models.items():
        s = survival_matrix(distribution, model.params, x_reference, months)[0]
        plt.plot(months, s, label=distribution)
    plt.xlabel("Month")
    plt.ylabel("Survival probability")
    plt.title("AFT survival curves by distribution")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_histogram(values, xlabel, title, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_boxplot(df, category, value, title, output_path):
    groups = [g[value].dropna().values for _, g in df.groupby(category)]
    labels = [str(k) for k in df.groupby(category).groups.keys()]
    plt.figure(figsize=(9, 5))
    plt.boxplot(groups, tick_labels=labels)
    plt.xlabel(category)
    plt.ylabel(value)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_bar(summary, x_col, y_col, title, output_path):
    summary = summary.sort_values(y_col, ascending=False)
    plt.figure(figsize=(9, 5))
    plt.bar(summary[x_col].astype(str), summary[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def segment_summary(df, columns, threshold, clv_col):
    out = []
    for col in columns:
        g = (
            df.groupby(col)
            .agg(
                n=("ID", "count"),
                observed_churn_rate=("event", "mean"),
                **{
                    f"mean_{clv_col}": (clv_col, "mean"),
                    f"median_{clv_col}": (clv_col, "median"),
                },
                mean_survival_12m=("survival_12m", "mean"),
                mean_churn_risk_12m=("churn_risk_12m", "mean"),
                at_risk_count=("churn_risk_12m", lambda x: int((x > threshold).sum())),
            )
            .reset_index()
            .rename(columns={col: "level"})
        )
        g.insert(0, "segment", col)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def format_amd(value):
    return f"{value:,.0f} AMD"


def coefficient_phrase(rows):
    if rows.empty:
        return "none"
    return ", ".join(
        f"`{row.term}` (time ratio {row.time_ratio:.2f})"
        for row in rows.itertuples(index=False)
    )


def write_report(output_path, comparison, final_coef, segment, budget, final_distribution, clv_months, clv_col):
    coef_core = final_coef[~final_coef["term"].isin(["Intercept", "log_sigma"])].copy()
    risk_terms = coef_core[coef_core["coef"] < 0].sort_values("time_ratio").head(3)
    protective_terms = coef_core[coef_core["coef"] > 0].sort_values("time_ratio", ascending=False).head(5)
    mean_clv_col = f"mean_{clv_col}"
    custcat = segment[segment["segment"].eq("custcat")].copy()
    valuable_segments = custcat.sort_values(
        [mean_clv_col, "mean_survival_12m"],
        ascending=[False, False],
    )
    riskiest_segment = custcat.sort_values("mean_churn_risk_12m", ascending=False).iloc[0]
    top_segments = valuable_segments.head(2)["level"].tolist()
    top_segments_text = " and ".join(top_segments)
    threshold_pct = budget["at_risk_threshold_churn_probability"] * 100
    expected_at_risk_churners = budget["expected_churners_12m_at_risk_group"]
    retention_budget = budget["suggested_upper_bound_retention_budget_AMD"]

    report = f"""# Homework 3: Survival Analysis Report

The final decision model is the reduced **{final_distribution} AFT model**, which had the lowest AIC after comparing the full AFT distributions and refitting statistically significant predictors (`outputs/model_comparison_with_final.csv`). In an AFT model, positive coefficients increase expected survival time and lower churn risk, while negative coefficients shorten expected survival time and raise churn risk. The main risk-increasing factors are {coefficient_phrase(risk_terms)}; the main protective factors are {coefficient_phrase(protective_terms)}. This means internet service, voice service, and being unmarried are associated with earlier churn, while older age, longer address history, and higher-service customer categories are associated with longer expected survival (`outputs/final_model_coefficients.csv`).

I define a valuable segment as one with high {clv_months}-month CLV and high 12-month survival probability, because it combines future margin with realistic retention potential. By this definition, the strongest customer-category segments are **{top_segments_text}**, while **{riskiest_segment["level"]}** has the highest average one-year churn risk and needs more retention attention (`outputs/segment_summary.csv`, `outputs/clv_per_customer.csv`). Assuming the data represents the full population, and defining at-risk subscribers as those with predicted 12-month churn probability above {threshold_pct:.0f}%, there are **{budget["at_risk_subscribers"]:,}** at-risk subscribers with **{expected_at_risk_churners:.2f}** expected churners within a year. I would set the annual retention budget at about **{format_amd(retention_budget)}**, calculated as at-risk CLV weighted by 12-month churn risk (`outputs/annual_retention_budget.csv`). For retention, I would prioritize high-CLV at-risk customers with internet or voice service, especially unmarried Basic or Total service customers, then use cheaper actions for lower-value risks: service-quality check-ins, plan education, bundle fixes, and controlled offer tests before scaling incentives.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telco.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--monthly-margin", type=float, default=1300.0)
    parser.add_argument("--annual-discount-rate", type=float, default=0.10)
    parser.add_argument("--clv-months", type=int, default=12)
    parser.add_argument("--at-risk-threshold", type=float, default=0.25)
    args = parser.parse_args()
    if args.clv_months < 12:
        parser.error("--clv-months must be at least 12 because the report uses 12-month churn risk.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    raw = pd.read_csv(args.data)
    data, x, scaler = prepare_data(raw)
    t = data["duration"].values
    e = data["event"].values

    xmat = np.column_stack([np.ones(len(x)), x.values])
    names = ["Intercept"] + x.columns.tolist()

    distributions = ["exponential", "weibull", "lognormal", "loglogistic"]
    full_models = {d: fit_aft(d, xmat, t, e, names) for d in distributions}

    comparison = pd.DataFrame(
        [
            {
                "model": k,
                "log_likelihood": v.log_likelihood,
                "n_params": len(v.params),
                "AIC": v.aic,
                "BIC": v.bic,
                "converged": v.success,
            }
            for k, v in full_models.items()
        ]
    ).sort_values("AIC")

    comparison.to_csv(os.path.join(args.output_dir, "model_comparison_full.csv"), index=False)

    best_full = comparison.iloc[0]["model"]
    best_coef = coefficient_table(full_models[best_full], best_full, xmat, t, e)
    best_coef.to_csv(os.path.join(args.output_dir, "best_full_model_coefficients.csv"), index=False)

    significant_terms = best_coef[
        (best_coef["term"] != "Intercept")
        & (best_coef["term"] != "log_sigma")
        & (best_coef["p_value"] < 0.05)
    ]["term"].tolist()

    x_final = x[significant_terms]
    x_final_mat = np.column_stack([np.ones(len(x_final)), x_final.values])
    final_names = ["Intercept"] + significant_terms

    final_distribution = best_full
    final_model = fit_aft(final_distribution, x_final_mat, t, e, final_names)
    final_coef = coefficient_table(final_model, final_distribution, x_final_mat, t, e)
    final_coef.to_csv(os.path.join(args.output_dir, "final_model_coefficients.csv"), index=False)

    final_row = pd.DataFrame(
        [
            {
                "model": f"final_{final_distribution}_significant_features",
                "log_likelihood": final_model.log_likelihood,
                "n_params": len(final_model.params),
                "AIC": final_model.aic,
                "BIC": final_model.bic,
                "converged": final_model.success,
            }
        ]
    )
    comparison_with_final = pd.concat([comparison, final_row], ignore_index=True).sort_values("AIC")
    comparison_with_final.to_csv(os.path.join(args.output_dir, "model_comparison_with_final.csv"), index=False)

    months = np.arange(1, args.clv_months + 1)
    survival = survival_matrix(final_distribution, final_model.params, x_final_mat, months)
    discounts = (1 + args.annual_discount_rate / 12) ** (months - 1)
    clv = args.monthly_margin * (survival / discounts).sum(axis=1)
    clv_col = f"CLV_{args.clv_months}m_AMD"

    result = raw.copy()
    result["event"] = e
    result["survival_12m"] = survival[:, 11]
    result["churn_risk_12m"] = 1 - result["survival_12m"]
    result[clv_col] = clv
    result["at_risk_12m"] = result["churn_risk_12m"] > args.at_risk_threshold

    for idx, month in enumerate(months, start=1):
        result[f"survival_m{month}"] = survival[:, idx - 1]

    result.to_csv(os.path.join(args.output_dir, "clv_per_customer.csv"), index=False)

    seg_cols = ["custcat", "internet", "marital", "gender", "region", "voice", "ed", "retire"]
    segments = segment_summary(result, seg_cols, args.at_risk_threshold, clv_col)
    segments.to_csv(os.path.join(args.output_dir, "segment_summary.csv"), index=False)

    expected_churners_12m = float(result["churn_risk_12m"].sum())
    at_risk = result[result["at_risk_12m"]]
    budget = {
        "at_risk_threshold_churn_probability": args.at_risk_threshold,
        "at_risk_subscribers": int(len(at_risk)),
        "expected_churners_12m_all_customers": expected_churners_12m,
        "expected_churners_12m_at_risk_group": float(at_risk["churn_risk_12m"].sum()),
        "sum_CLV_at_risk_group_AMD": float(at_risk[clv_col].sum()),
        "suggested_upper_bound_retention_budget_AMD": float((at_risk[clv_col] * at_risk["churn_risk_12m"]).sum()),
    }
    pd.DataFrame([budget]).to_csv(os.path.join(args.output_dir, "annual_retention_budget.csv"), index=False)

    reference = np.zeros((1, xmat.shape[1]))
    reference[0, 0] = 1.0
    plot_model_curves(full_models, reference, months, os.path.join(args.output_dir, "plots", "aft_survival_curves_all_models.png"))
    plot_histogram(result[clv_col], f"{args.clv_months}-month CLV, AMD", "CLV distribution", os.path.join(args.output_dir, "plots", "clv_distribution.png"))
    plot_boxplot(result, "custcat", clv_col, "CLV by customer category", os.path.join(args.output_dir, "plots", "clv_by_custcat.png"))
    plot_bar(
        segments[segments["segment"].eq("custcat")],
        "level",
        "mean_churn_risk_12m",
        "Mean 12-month churn risk by customer category",
        os.path.join(args.output_dir, "plots", "churn_risk_by_custcat.png"),
    )

    write_report(
        os.path.join(args.output_dir, "report.md"),
        comparison_with_final,
        final_coef,
        segments,
        budget,
        final_distribution,
        args.clv_months,
        clv_col,
    )

    with open(os.path.join(args.output_dir, "scaler.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(scaler, f, indent=2)

    print("Done. Outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
