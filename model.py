import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# Load data & mean-center continuous predictors
df = pd.read_csv("master_tableA.csv", parse_dates=["Timestamp"])

model_cols = [
    "Heart_Rate_BPM", "Breathing_Rate_BPM",
    "Box_Temp_C", "Outdoor_Pressure_hPa", "Box_Humidity_pct", "Light_Lux",
    "State", "Individual", "Recording", "Intervention_Day", "Timestamp",
]

df_clean = df.dropna(subset=[c for c in model_cols if c != "Individual"]).copy()
df_clean["Intervention_Day"] = df_clean["Intervention_Day"].astype(int)
df_clean = df_clean.sort_values("Timestamp").reset_index(drop=True)

# Log-transform skewed light variable
df_clean["log_Light_Lux"] = np.log1p(df_clean["Light_Lux"].clip(lower=0))

# Mean-center continuous variables across full dataset for common baseline reference
continuous_vars = ["Box_Temp_C", "Outdoor_Pressure_hPa", "Box_Humidity_pct", "log_Light_Lux"]
for col in continuous_vars:
    df_clean[f"{col}_c"] = df_clean[col] - df_clean[col].mean()

for col in continuous_vars:
    df_clean[f"{col}_c"] = df_clean[col] - df_clean[col].mean()

#  Split dataset into State subsets
df_hib = df_clean[df_clean["State"] == "hibernating"].copy()
df_aro = df_clean[df_clean["State"] == "aroused"].copy()

print(f"Data split: Hibernating (N = {len(df_hib)}), Aroused (N = {len(df_aro)})")

# Simple additive predictor formula (no state terms or interactions required)
predictors_formula = "Box_Temp_C_c + Outdoor_Pressure_hPa_c + Box_Humidity_pct_c + log_Light_Lux_c + Intervention_Day"


# 3. Fit 4 State-Stratified Models (Cluster-robust SE by Recording)
models = {
    ("Heart Rate", "Hibernating"): smf.ols(f"Heart_Rate_BPM ~ {predictors_formula}", data=df_hib).fit(
        cov_type="cluster", cov_kwds={"groups": df_hib["Recording"]}
    ),
    ("Heart Rate", "Aroused"): smf.ols(f"Heart_Rate_BPM ~ {predictors_formula}", data=df_aro).fit(
        cov_type="cluster", cov_kwds={"groups": df_aro["Recording"]}
    ),
    ("Respiration Rate", "Hibernating"): smf.ols(f"Breathing_Rate_BPM ~ {predictors_formula}", data=df_hib).fit(
        cov_type="cluster", cov_kwds={"groups": df_hib["Recording"]}
    ),
    ("Respiration Rate", "Aroused"): smf.ols(f"Breathing_Rate_BPM ~ {predictors_formula}", data=df_aro).fit(
        cov_type="cluster", cov_kwds={"groups": df_aro["Recording"]}
    )
}

# Attach fitted values back for evaluation and plotting
df_hib["HR_Pred"] = models[("Heart Rate", "Hibernating")].fittedvalues
df_aro["HR_Pred"] = models[("Heart Rate", "Aroused")].fittedvalues
df_hib["BR_Pred"] = models[("Respiration Rate", "Hibernating")].fittedvalues
df_aro["BR_Pred"] = models[("Respiration Rate", "Aroused")].fittedvalues

df_clean_eval = pd.concat([df_hib, df_aro]).sort_index()


# Summary Coefficients Table

summary_rows = []
for (outcome, state), res in models.items():
    for var in res.params.index:
        summary_rows.append({
            "Outcome": outcome,
            "State": state,
            "Variable": var,
            "Coef": res.params[var],
            "SE": res.bse[var],
            "p-value": res.pvalues[var],
            "R2": res.rsquared
        })

summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 80)
print("STATE-STRATIFIED OLS MODEL COEFFICIENTS & WITHIN-STATE R^2")
print("=" * 80)
print(summary_df.round(4).to_string(index=False))

# PLOT 1: Observed vs Predicted by State

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# Heart Rate Fit
sns.scatterplot(data=df_clean_eval, x="HR_Pred", y="Heart_Rate_BPM", hue="State",
                palette="Set2", alpha=0.6, s=25, ax=axes[0])
lims_hr = [df_clean_eval[["HR_Pred", "Heart_Rate_BPM"]].min().min(),
           df_clean_eval[["HR_Pred", "Heart_Rate_BPM"]].max().max()]
axes[0].plot(lims_hr, lims_hr, "k--", alpha=0.7, linewidth=1.5, label="1:1 Identity")
axes[0].set_title(
    f"Heart Rate Fit\nHib R² = {models[('Heart Rate', 'Hibernating')].rsquared:.3f} | Aro R² = {models[('Heart Rate', 'Aroused')].rsquared:.3f}",
    fontweight="bold", fontsize=11
)
axes[0].set_xlabel("Predicted Heart Rate (BPM)")
axes[0].set_ylabel("Observed Heart Rate (BPM)")
axes[0].legend(loc="upper left")

# Respiration Rate Fit
sns.scatterplot(data=df_clean_eval, x="BR_Pred", y="Breathing_Rate_BPM", hue="State",
                palette="Set2", alpha=0.6, s=25, ax=axes[1])
lims_br = [df_clean_eval[["BR_Pred", "Breathing_Rate_BPM"]].min().min(),
           df_clean_eval[["BR_Pred", "Breathing_Rate_BPM"]].max().max()]
axes[1].plot(lims_br, lims_br, "k--", alpha=0.7, linewidth=1.5, label="1:1 Identity")
axes[1].set_title(
    f"Respiration Rate Fit\nHib R² = {models[('Respiration Rate', 'Hibernating')].rsquared:.3f} | Aro R² = {models[('Respiration Rate', 'Aroused')].rsquared:.3f}",
    fontweight="bold", fontsize=11
)
axes[1].set_xlabel("Predicted Breathing Rate (BPM)")
axes[1].set_ylabel("Observed Breathing Rate (BPM)")
axes[1].legend(loc="upper left")

plt.tight_layout()
plt.savefig("stratified_observed_vs_predicted.png", bbox_inches="tight")
plt.show()


# PLOT 2: Forest Plot Comparing States Side-by-Side

def extract_stratified_coefs(model_dict):
    rows = []
    for (outcome, state), res in model_dict.items():
        coefs = res.params.drop("Intercept")
        conf = res.conf_int().loc[coefs.index]
        for var in coefs.index:
            rows.append({
                "Outcome": outcome,
                "State": state,
                "Variable": var,
                "Estimate": coefs[var],
                "Lower_CI": conf.loc[var, 0],
                "Upper_CI": conf.loc[var, 1],
            })
    return pd.DataFrame(rows)

coef_df = extract_stratified_coefs(models)
all_vars = list(dict.fromkeys(coef_df["Variable"].tolist()))
var_to_ypos = {v: i for i, v in enumerate(all_vars)}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharey=True)
outcomes = ["Heart Rate", "Respiration Rate"]

palette = {"Hibernating": "#66c2a5", "Aroused": "#fc8d62"}
offsets = {"Hibernating": -0.12, "Aroused": 0.12}

for i, outcome in enumerate(outcomes):
    ax = axes[i]
    sub_df = coef_df[coef_df["Outcome"] == outcome]
    
    for state in ["Hibernating", "Aroused"]:
        st_df = sub_df[sub_df["State"] == state].copy()
        st_df["y"] = st_df["Variable"].map(var_to_ypos) + offsets[state]
        
        ax.errorbar(
            x=st_df["Estimate"], y=st_df["y"],
            xerr=[st_df["Estimate"] - st_df["Lower_CI"], st_df["Upper_CI"] - st_df["Estimate"]],
            fmt="o", color=palette[state], ecolor=palette[state], elinewidth=1.5, capsize=4, label=state
        )
    
    ax.axvline(0, color="black", linestyle="--", alpha=0.6, linewidth=1)
    ax.set_title(f"{outcome} Environmental Effects", fontweight="bold")
    ax.set_xlabel(r"Regression Coefficient ($\beta$) with 95% Cluster-Robust CI")
    ax.legend(loc="best")
    if i == 0:
        ax.set_yticks(range(len(all_vars)))
        ax.set_yticklabels(all_vars, fontsize=9)
        ax.invert_yaxis()

plt.tight_layout()
plt.savefig("stratified_coefficients_forest_plot.png", bbox_inches="tight")
plt.show()


STATE_PALETTE = {
    "hibernating": "#66c2a5",  # Light green / teal
    "aroused": "#fc8d62"       # Light salmon / orange
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

outcomes = [
    ("Heart_Rate_BPM", "HR_Pred", "Heart Rate"),
    ("Breathing_Rate_BPM", "BR_Pred", "Respiration Rate")
]

for col_idx, (y_col, x_col, label) in enumerate(outcomes):
    for row_idx, state in enumerate(["hibernating", "aroused"]):
        ax = axes[row_idx, col_idx]
        sub_df = df_clean_eval[df_clean_eval["State"] == state]
        
        # Scatter plot for single state
        sns.scatterplot(
            data=sub_df, x=x_col, y=y_col, 
            color=STATE_PALETTE[state], alpha=0.4, s=20, ax=ax
        )
        
        # 1:1 Identity line
        lims = [min(sub_df[x_col].min(), sub_df[y_col].min()),
                max(sub_df[x_col].max(), sub_df[y_col].max())]
        ax.plot(lims, lims, "k--", alpha=0.7, label="1:1 Identity")
        
        ax.set_title(f"{label} - {state.capitalize()}")
        ax.set_xlabel(f"Predicted {label}")
        ax.set_ylabel(f"Observed {label}")

plt.tight_layout()
plt.savefig("seperated_plot.png", bbox_inches="tight")

plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Unified color palette matching state strings in your dataframe
STATE_PALETTE = {
    "hibernating": "#66c2a5",  # Light green / teal
    "aroused": "#fc8d62"       # Light salmon / orange
}

# Create a 2x2 grid: Top row = Breathing Rate, Bottom row = Heart Rate
fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

states = ["hibernating", "aroused"]

# 1. Top Row: Temperature vs Breathing Rate
for col_idx, state in enumerate(states):
    ax = axes[0, col_idx]
    sub_df = df_clean[df_clean["State"] == state]
    color = STATE_PALETTE[state]
    
    # Scatter points
    sns.scatterplot(
        data=sub_df, x="Box_Temp_C", y="Breathing_Rate_BPM",
        color=color, alpha=0.4, s=25, ax=ax
    )
    
    # Add linear trendline
    sns.regplot(
        data=sub_df, x="Box_Temp_C", y="Breathing_Rate_BPM",
        ax=ax, scatter=False, color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
    )
    
    ax.set_title(f"Breathing Rate vs Temp ({state.capitalize()})", fontweight="bold")
    ax.set_xlabel("Box Temperature (°C)")
    ax.set_ylabel("Breathing Rate (BPM)")

# 2. Bottom Row: Temperature vs Heart Rate
for col_idx, state in enumerate(states):
    ax = axes[1, col_idx]
    sub_df = df_clean[df_clean["State"] == state]
    color = STATE_PALETTE[state]
    
    # Scatter points
    sns.scatterplot(
        data=sub_df, x="Box_Temp_C", y="Heart_Rate_BPM",
        color=color, alpha=0.4, s=25, ax=ax
    )
    
    # Add linear trendline
    sns.regplot(
        data=sub_df, x="Box_Temp_C", y="Heart_Rate_BPM",
        ax=ax, scatter=False, color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
    )
    
    ax.set_title(f"Heart Rate vs Temp ({state.capitalize()})", fontweight="bold")
    ax.set_xlabel("Box Temperature (°C)")
    ax.set_ylabel("Heart Rate (BPM)")

plt.tight_layout()
plt.savefig("temperature_vs_vitals_by_state.png", bbox_inches="tight")
plt.show()


#####


# Create cross-tabulation of State vs Intervention Day
ct = (
    pd.crosstab(df_clean["Intervention_Day"], df_clean["State"], normalize="index")
    * 100
)

# 2. Set up the figure and axis with dpi explicitly specified here
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

# 3. Plot without passing dpi into ct.plot()
ct.plot(kind="bar", stacked=True, color=["#fc8d62", "#66c2a5"], ax=ax)

plt.title("Proportion of Time in State by Intervention Day", fontweight="bold")
plt.xlabel("Intervention Day")
plt.ylabel("Percentage of Time (%)")
plt.legend(title="State", loc="upper right")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("intervention_vs_state_bar.png")
plt.show()


# -------------------------------------------------------------------------
# Box Humidity (%) vs Vitals
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

states = ["hibernating", "aroused"]

# Top Row: Breathing Rate vs Humidity
for col_idx, state in enumerate(states):
    ax = axes[0, col_idx]
    sub_df = df_clean[df_clean["State"] == state]
    color = STATE_PALETTE[state]
    
    sns.scatterplot(
        data=sub_df, x="Box_Humidity_pct", y="Breathing_Rate_BPM",
        color=color, alpha=0.4, s=25, ax=ax
    )
    sns.regplot(
        data=sub_df, x="Box_Humidity_pct", y="Breathing_Rate_BPM",
        ax=ax, scatter=False, color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
    )
    ax.set_title(f"Breathing Rate vs Humidity ({state.capitalize()})", fontweight="bold")
    ax.set_xlabel("Box Humidity (%)")
    ax.set_ylabel("Breathing Rate (BPM)")

# Bottom Row: Heart Rate vs Humidity
for col_idx, state in enumerate(states):
    ax = axes[1, col_idx]
    sub_df = df_clean[df_clean["State"] == state]
    color = STATE_PALETTE[state]
    
    sns.scatterplot(
        data=sub_df, x="Box_Humidity_pct", y="Heart_Rate_BPM",
        color=color, alpha=0.4, s=25, ax=ax
    )
    sns.regplot(
        data=sub_df, x="Box_Humidity_pct", y="Heart_Rate_BPM",
        ax=ax, scatter=False, color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
    )
    ax.set_title(f"Heart Rate vs Humidity ({state.capitalize()})", fontweight="bold")
    ax.set_xlabel("Box Humidity (%)")
    ax.set_ylabel("Heart Rate (BPM)")

plt.tight_layout()
plt.savefig("humidity_vs_vitals_by_state.png", bbox_inches="tight")
plt.show()


# -------------------------------------------------------------------------
# Outdoor Pressure (hPa) vs Vitals
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

# Top Row: Breathing Rate vs Outdoor Pressure
for col_idx, state in enumerate(states):
    ax = axes[0, col_idx]
    sub_df = df_clean[df_clean["State"] == state]
    color = STATE_PALETTE[state]
    
    sns.scatterplot(
        data=sub_df, x="Outdoor_Pressure_hPa", y="Breathing_Rate_BPM",
        color=color, alpha=0.4, s=25, ax=ax
    )
    sns.regplot(
        data=sub_df, x="Outdoor_Pressure_hPa", y="Breathing_Rate_BPM",
        ax=ax, scatter=False, color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
    )
    ax.set_title(f"Breathing Rate vs Pressure ({state.capitalize()})", fontweight="bold")
    ax.set_xlabel("Outdoor Pressure (hPa)")
    ax.set_ylabel("Breathing Rate (BPM)")

# Bottom Row: Heart Rate vs Outdoor Pressure
for col_idx, state in enumerate(states):
    ax = axes[1, col_idx]
    sub_df = df_clean[df_clean["State"] == state]
    color = STATE_PALETTE[state]
    
    sns.scatterplot(
        data=sub_df, x="Outdoor_Pressure_hPa", y="Heart_Rate_BPM",
        color=color, alpha=0.4, s=25, ax=ax
    )
    sns.regplot(
        data=sub_df, x="Outdoor_Pressure_hPa", y="Heart_Rate_BPM",
        ax=ax, scatter=False, color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
    )
    ax.set_title(f"Heart Rate vs Pressure ({state.capitalize()})", fontweight="bold")
    ax.set_xlabel("Outdoor Pressure (hPa)")
    ax.set_ylabel("Heart Rate (BPM)")

plt.tight_layout()
plt.savefig("pressure_vs_vitals_by_state.png", bbox_inches="tight")
plt.show()


# -------------------------------------------------------------------------
# Log Light (log1p Lux) vs Vitals
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

# Top Row: Breathing Rate vs Log Light
for col_idx, state in enumerate(states):
    ax = axes[0, col_idx]
    sub_df = df_clean[df_clean["State"] == state]
    color = STATE_PALETTE[state]
    
    sns.scatterplot(
        data=sub_df, x="log_Light_Lux", y="Breathing_Rate_BPM",
        color=color, alpha=0.4, s=25, ax=ax
    )
    sns.regplot(
        data=sub_df, x="log_Light_Lux", y="Breathing_Rate_BPM",
        ax=ax, scatter=False, color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
    )
    ax.set_title(f"Breathing Rate vs Log Light ({state.capitalize()})", fontweight="bold")
    ax.set_xlabel("Log Light (log1p Lux)")
    ax.set_ylabel("Breathing Rate (BPM)")

# Bottom Row: Heart Rate vs Log Light
for col_idx, state in enumerate(states):
    ax = axes[1, col_idx]
    sub_df = df_clean[df_clean["State"] == state]
    color = STATE_PALETTE[state]
    
    sns.scatterplot(
        data=sub_df, x="log_Light_Lux", y="Heart_Rate_BPM",
        color=color, alpha=0.4, s=25, ax=ax
    )
    sns.regplot(
        data=sub_df, x="log_Light_Lux", y="Heart_Rate_BPM",
        ax=ax, scatter=False, color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
    )
    ax.set_title(f"Heart Rate vs Log Light ({state.capitalize()})", fontweight="bold")
    ax.set_xlabel("Log Light (log1p Lux)")
    ax.set_ylabel("Heart Rate (BPM)")

plt.tight_layout()
plt.savefig("light_vs_vitals_by_state.png", bbox_inches="tight")
plt.show()



env_vars = [
    ("Box_Humidity_pct", "Box Humidity (%)", "humidity"),
    ("Outdoor_Pressure_hPa", "Outdoor Pressure (hPa)", "pressure"),
    ("log_Light_Lux", "Log Light (log1p Lux)", "light")
]

for col_name, label_name, file_prefix in env_vars:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    
    for row_idx, (vital_col, vital_label) in enumerate([("Breathing_Rate_BPM", "Breathing Rate"), ("Heart_Rate_BPM", "Heart Rate")]):
        for col_idx, state in enumerate(["hibernating", "aroused"]):
            ax = axes[row_idx, col_idx]
            sub_df = df_clean[df_clean["State"] == state]
            color = STATE_PALETTE[state]
            
            sns.scatterplot(
                data=sub_df, x=col_name, y=vital_col,
                color=color, alpha=0.4, s=25, ax=ax
            )
            sns.regplot(
                data=sub_df, x=col_name, y=vital_col,
                ax=ax, scatter=False, color="black", line_kws={"linewidth": 1.5, "linestyle": "--"}
            )
            ax.set_title(f"{vital_label} vs {label_name} ({state.capitalize()})", fontweight="bold")
            ax.set_xlabel(label_name)
            ax.set_ylabel(f"{vital_label} (BPM)")

    plt.tight_layout()
    plt.savefig(f"{file_prefix}_vs_vitals_by_state.png", bbox_inches="tight")
    plt.close()