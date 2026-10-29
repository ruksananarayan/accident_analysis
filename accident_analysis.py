"""
Analysis of Public Industrial Accident Data: Trends, Frequency and Severity Rates
23ME2044 Industrial Safety Engineering - Study Project

USAGE
  python accident_analysis.py                    -> DEMO mode (synthetic data, same column names as the
                                                    OSHA Severe Injury Reports file)
  python accident_analysis.py --input real.csv   -> REAL mode (OSHA Severe Injury Reports CSV)

Outputs: figs/*.png and stats.json (used to build the report tables/figures).
The demo data is SYNTHETIC. It exists only to test the pipeline. Replace it with the real
open dataset before submission and re-write the discussion to match the real numbers.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=None)
ap.add_argument("--out", default="figs")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(2026)

SECTORS = ["Manufacturing", "Construction", "Transport & Warehousing",
           "Retail Trade", "Health Care", "Agriculture"]
# model exposure (million man-hours in year 2018) and base injury rate per million man-hours
HOURS0 = dict(zip(SECTORS, [90, 60, 40, 70, 80, 15]))
RATE0 = dict(zip(SECTORS, [8.0, 14.0, 11.0, 3.0, 2.5, 12.0]))
EVENTS = ["Struck by object", "Fall from elevation", "Caught in / between", "Overexertion",
          "Contact with electricity", "Fire / explosion", "Exposure to harmful substance",
          "Transportation incident"]
EV_W = {
    "Manufacturing":           [.15, .08, .35, .10, .05, .07, .10, .10],
    "Construction":            [.20, .35, .12, .05, .12, .04, .04, .08],
    "Transport & Warehousing": [.18, .10, .12, .15, .02, .03, .05, .35],
    "Retail Trade":            [.22, .15, .08, .30, .03, .02, .05, .15],
    "Health Care":             [.12, .10, .03, .45, .02, .02, .20, .06],
    "Agriculture":             [.15, .12, .20, .10, .04, .03, .08, .28],
}
NATURE = ["Fracture", "Amputation", "Laceration", "Burn", "Head injury", "Crushing injury", "Other"]
YEARS = list(range(2018, 2025))

def make_demo():
    rows = []
    for y in YEARS:
        for s in SECTORS:
            hrs = HOURS0[s] * (1 + 0.01 * (y - 2018))
            rate = RATE0[s] * (0.97 ** (y - 2018)) * (0.85 if y == 2020 else 1.0)
            n = rng.poisson(hrs * rate)
            ev = rng.choice(EVENTS, size=n, p=EV_W[s])
            mon = rng.choice(range(1, 13), size=n, p=np.array([.07,.07,.08,.08,.09,.10,.10,.10,.09,.09,.07,.06]))
            day = rng.integers(1, 28, size=n)
            for e, m, d in zip(ev, mon, day):
                if e == "Caught in / between":
                    nat = rng.choice(NATURE, p=[.15,.40,.10,.00,.02,.28,.05])
                elif e == "Fire / explosion":
                    nat = rng.choice(NATURE, p=[.05,.02,.05,.70,.08,.05,.05])
                elif e == "Fall from elevation":
                    nat = rng.choice(NATURE, p=[.55,.01,.03,.00,.25,.10,.06])
                else:
                    nat = rng.choice(NATURE, p=[.30,.06,.28,.04,.08,.10,.14])
                amp = 1 if nat == "Amputation" else 0
                hosp = 1 if (amp or rng.random() < {"Fall from elevation": .85, "Fire / explosion": .8,
                             "Caught in / between": .75}.get(e, .55)) else 0
                days = int(rng.lognormal(3.0 if hosp else 2.0, 0.8)) + (600 if rng.random() < 0.004 else 0)
                rows.append((f"{y}-{m:02d}-{d:02d}", s, e, nat, hosp, amp, days))
    df = pd.DataFrame(rows, columns=["EventDate", "Sector", "EventTitle", "NatureTitle",
                                     "Hospitalized", "Amputation", "DaysLost"])
    # inject typical data-quality problems
    dup = df.sample(frac=0.02, random_state=1)
    df = pd.concat([df, dup], ignore_index=True)
    idx = df.sample(frac=0.015, random_state=2).index
    df.loc[idx, "EventTitle"] = np.nan
    idx = df.sample(frac=0.03, random_state=3).index
    df.loc[idx, "EventTitle"] = df.loc[idx, "EventTitle"].astype(str).str.upper().replace("NAN", np.nan)
    idx = df.sample(frac=0.02, random_state=4).index
    df.loc[idx, "Sector"] = df.loc[idx, "Sector"] + " "
    idx = df.sample(frac=0.004, random_state=5).index
    df.loc[idx, "EventDate"] = "2031-01-01"          # impossible future dates
    return df

if args.input:
    df = pd.read_csv(args.input, low_memory=False)
    # map OSHA Severe Injury column names to the working schema
    df = df.rename(columns={"Primary NAICS": "Sector"}) if "Primary NAICS" in df.columns else df
    if "DaysLost" not in df.columns:
        df["DaysLost"] = np.nan   # not published in the OSHA file -> severity rate not computed
    MODE = "real"
else:
    df = make_demo()
    MODE = "demo"

stats = {"mode": MODE, "n_raw": int(len(df))}
log = []

# ---------------- cleaning ----------------
n0 = len(df)
df = df.drop_duplicates(); log.append(("Exact duplicate records removed", n0 - len(df))); n0 = len(df)
df["EventDate"] = pd.to_datetime(df["EventDate"], errors="coerce")
bad = df["EventDate"].isna() | (df["EventDate"] > pd.Timestamp.today())
df = df[~bad]; log.append(("Invalid or future event dates removed", n0 - len(df))); n0 = len(df)
for c in ["Sector", "EventTitle", "NatureTitle"]:
    df[c] = df[c].astype("string").str.strip().str.title()
df["EventTitle"] = df["EventTitle"].replace({"Struck By Object": "Struck by object",
    "Fall From Elevation": "Fall from elevation", "Caught In / Between": "Caught in / between",
    "Contact With Electricity": "Contact with electricity", "Fire / Explosion": "Fire / explosion",
    "Exposure To Harmful Substance": "Exposure to harmful substance"})
df["Sector"] = df["Sector"].replace({"Transport & Warehousing": "Transport & Warehousing"})
miss = int(df["EventTitle"].isna().sum())
df = df.dropna(subset=["EventTitle"]); log.append(("Records with missing event type removed", miss))
df["Year"] = df["EventDate"].dt.year; df["Month"] = df["EventDate"].dt.month
stats["log"] = log; stats["n_clean"] = int(len(df))
stats["years"] = [int(df.Year.min()), int(df.Year.max())]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
C = ["#1F3A5F", "#C0392B", "#2E86AB", "#E39B2D", "#5B8C5A", "#7D5BA6", "#888888", "#A0522D"]

# ---------------- exposure & rates (demo exposure table) ----------------
yrs = sorted(df.Year.unique())
hours = pd.DataFrame({s: [HOURS0[s] * (1 + 0.01 * (y - 2018)) for y in yrs] for s in SECTORS}, index=yrs)
by_ys = df.groupby(["Year", "Sector"]).agg(cases=("EventTitle", "size"), days=("DaysLost", "sum")).reset_index()
by_ys["hours_m"] = by_ys.apply(lambda r: hours.loc[r.Year, r.Sector] if r.Sector in hours.columns else np.nan, axis=1)
by_ys["FR"] = by_ys.cases / by_ys.hours_m           # per million man-hours
by_ys["SR"] = by_ys.days / by_ys.hours_m
tot = by_ys.groupby("Year").agg(cases=("cases", "sum"), days=("days", "sum"), hours_m=("hours_m", "sum"))
tot["FR"] = tot.cases / tot.hours_m; tot["SR"] = tot.days / tot.hours_m
sec = by_ys.groupby("Sector").agg(cases=("cases", "sum"), days=("days", "sum"), hours_m=("hours_m", "sum"))
sec["FR"] = sec.cases / sec.hours_m; sec["SR"] = sec.days / sec.hours_m
sec = sec.sort_values("FR", ascending=False)
stats["yearly"] = {int(y): dict(cases=int(r.cases), FR=round(r.FR, 2), SR=round(r.SR, 1)) for y, r in tot.iterrows()}
stats["sector"] = {s: dict(cases=int(r.cases), FR=round(r.FR, 2), SR=round(r.SR, 1)) for s, r in sec.iterrows()}
stats["hours_total_m"] = round(float(tot.hours_m.sum()), 1)

# Fig 5.1 trend
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.bar(tot.index, tot.cases, color=C[2], width=.6, label="Reported cases")
ax.set_ylabel("Number of severe injury cases"); ax.set_xlabel("Year")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(tot.index, tot.FR, color=C[1], marker="o", label="Frequency rate")
ax2.set_ylabel("Frequency rate (cases per 10$^6$ man-hours)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{args.out}/fig5_1.png", dpi=200); plt.close()

# Fig 5.2 FR vs SR by sector
fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.3), sharey=True)
y = np.arange(len(sec))
axs[0].barh(y, sec.FR, color=C[0]); axs[0].set_yticks(y); axs[0].set_yticklabels(sec.index)
axs[0].invert_yaxis(); axs[0].set_xlabel("Frequency rate\n(cases per 10$^6$ man-hours)")
axs[1].barh(y, sec.SR, color=C[1]); axs[1].set_xlabel("Severity rate\n(man-days lost per 10$^6$ man-hours)")
for a in axs: a.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(f"{args.out}/fig5_2.png", dpi=200); plt.close()

# Fig 5.3 Pareto of events
ev = df.EventTitle.value_counts(); cum = ev.cumsum() / ev.sum() * 100
stats["events"] = {k: dict(n=int(v), pct=round(v / ev.sum() * 100, 1)) for k, v in ev.items()}
stats["pareto_top3_pct"] = round(float(cum.iloc[2]), 1)
fig, ax = plt.subplots(figsize=(6.4, 3.5))
ax.bar(range(len(ev)), ev.values, color=C[0])
ax.set_xticks(range(len(ev))); ax.set_xticklabels(ev.index, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Number of cases"); ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(range(len(ev)), cum.values, color=C[1], marker="o"); ax2.set_ylim(0, 105)
ax2.set_ylabel("Cumulative share (%)")
fig.tight_layout(); fig.savefig(f"{args.out}/fig5_3.png", dpi=200); plt.close()

# Fig 5.4 Monthly profile
mon = df.groupby("Month").size()
stats["peak_month"] = int(mon.idxmax()); stats["low_month"] = int(mon.idxmin())
stats["peak_vs_low_pct"] = round(float((mon.max() / mon.min() - 1) * 100), 1)
fig, ax = plt.subplots(figsize=(6.2, 3.1))
ax.plot(mon.index, mon.values, marker="o", color=C[1]); ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_ylabel("Cases (all years)"); ax.set_xlabel("Month"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{args.out}/fig5_4.png", dpi=200); plt.close()

# Fig 5.5 severity outcome by event type
g = df.groupby("EventTitle").agg(n=("Hospitalized", "size"), hosp=("Hospitalized", "mean"), amp=("Amputation", "mean"))
g = g.sort_values("hosp", ascending=False)
stats["outcome"] = {k: dict(hosp=round(r.hosp * 100, 1), amp=round(r.amp * 100, 1)) for k, r in g.iterrows()}
stats["hosp_overall"] = round(float(df.Hospitalized.mean() * 100), 1)
stats["amp_overall"] = round(float(df.Amputation.mean() * 100), 1)
fig, ax = plt.subplots(figsize=(6.4, 3.4)); x = np.arange(len(g)); w = .38
ax.bar(x - w/2, g.hosp * 100, w, color=C[0], label="Hospitalised (%)")
ax.bar(x + w/2, g.amp * 100, w, color=C[3], label="Amputation (%)")
ax.set_xticks(x); ax.set_xticklabels(g.index, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Share of cases (%)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{args.out}/fig5_5.png", dpi=200); plt.close()

# trend test (least-squares slope of FR vs year)
slope = np.polyfit(tot.index, tot.FR, 1)[0]
stats["FR_slope_per_year"] = round(float(slope), 3)
stats["FR_change_pct"] = round(float((tot.FR.iloc[-1] / tot.FR.iloc[0] - 1) * 100), 1)
# severity per case by sector
stats["days_per_case"] = round(float(df.DaysLost.mean()), 1) if df.DaysLost.notna().any() else None

json.dump(stats, open("stats.json", "w"), indent=1)
df.to_csv("cleaned_data.csv", index=False)
print(json.dumps(stats, indent=1)[:1800])
