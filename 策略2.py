import pandas as pd
import numpy as np
import re
from collections import Counter

import plt

df = pd.read_csv("the_ordinary_reviews.csv", low_memory=False)
df = df[["ingredients", "review_text", "rating_x", "product_name_x"]].copy()

df["ingredients"] = df["ingredients"].fillna("").astype(str)
df["review_text"] = df["review_text"].fillna("").astype(str)

print("Total reviews:", len(df))


def normalize_ingredients_text(x: str) -> str:
    x = x.lower()

    x = x.replace("[", " ").replace("]", " ")
    x = x.replace("{", " ").replace("}", " ")
    x = x.replace("(", " ").replace(")", " ")
    x = x.replace('"', " ").replace("'", " ")


    x = x.replace("\n", " ").replace("\r", " ")
    x = x.replace(";", ",")
    x = x.replace("|", ",")

    x = re.sub(r"\s+", " ", x).strip()

    return x

df["ingredients_clean_text"] = df["ingredients"].apply(normalize_ingredients_text)



def tokenize_ingredients(clean_text: str):
    parts = [p.strip() for p in clean_text.split(",")]
    tokens = []
    for p in parts:
        if not p:
            continue


        p = re.sub(r"[\.]+$", "", p).strip()


        if re.fullmatch(r"\d+", p):
            continue
        if len(p) <= 1:
            continue

        tokens.append(p)
    return tokens

df["ingredient_tokens_raw"] = df["ingredients_clean_text"].apply(tokenize_ingredients)



STOP_EXACT = set([

    "water", "aqua", "aqua water",

    "glycerin", "propanediol", "propylene glycol", "butylene glycol", "pentylene glycol",
    "caprylyl glycol", "1,2-hexanediol", "2-hexanediol", "hexanediol", "ethylhexylglycerin",
    "ethoxydiglycol", "dimethyl isosorbide",


    "phenoxyethanol", "chlorphenesin", "sodium benzoate", "potassium sorbate",


    "citric acid", "sodium hydroxide", "trisodium ethylenediamine disuccinate",
    "disodium edta", "tetrasodium edta", "sodium chloride",


    "xanthan gum", "sclerotium gum", "tamarindus indica seed gum",


    "dimethicone", "polysilicone-11", "isohexadecane",
])

STOP_KEYWORDS = [
    "water", "aqua",
    "glycol", "hexanediol",
    "phenoxyethanol", "chlorphenesin",
    "edta",
]

def is_stop_token(tok: str) -> bool:
    t = tok.strip()
    if t in STOP_EXACT:
        return True
    for kw in STOP_KEYWORDS:
        if kw in t:
            return True
    return False

def filter_stop(tokens):
    return [t for t in tokens if not is_stop_token(t)]

df["active_ingredients"] = df["ingredient_tokens_raw"].apply(filter_stop)

sample_active = df["active_ingredients"].dropna().iloc[0]
print("\nExample active ingredients (after stop-filter):")
print(sample_active)


SYMPTOM_WORDS = [
    "burn", "burning", "stinging", "sting",
    "irritat", "irritation",  # Use "irritat" tocover irritate/irritated/irritation
    "redness", "red", "rash",
    "breakout", "breakouts", "broke out",
    "acne", "pimples", "pimple", "cystic",
    "itch", "itchy", "peeling", "peel", "flaking"
]
symptom_pattern = re.compile("|".join(SYMPTOM_WORDS), flags=re.IGNORECASE)

df["symptom_flag"] = df["review_text"].apply(lambda x: bool(symptom_pattern.search(x)))

# Option A
df["problem_A"] = df["symptom_flag"]

# Option B
df["problem_B"] = df["symptom_flag"] & (df["rating_x"].astype(float) <= 2)

print("\nProblem counts:")
print("A (symptom only):", int(df["problem_A"].sum()))
print("B (symptom + rating<=2):", int(df["problem_B"].sum()))


def compute_risk_ratio(problem_col: str, min_problem_count=30, min_total_count=50, smoothing=0.5):

    prob_df = df[df[problem_col]].copy()
    non_df  = df[~df[problem_col]].copy()

    def review_level_counter(sub_df):
        c = Counter()
        for lst in sub_df["active_ingredients"]:
            unique_ings = set(lst)
            c.update(unique_ings)
        return c

    prob_counts = review_level_counter(prob_df)
    non_counts  = review_level_counter(non_df)

    n_prob = len(prob_df)
    n_non  = len(non_df)

    rows = []
    for ing, cnt_p in prob_counts.items():
        cnt_n = non_counts.get(ing, 0)
        cnt_total = cnt_p + cnt_n

        if cnt_p < min_problem_count:
            continue
        if cnt_total < min_total_count:
            continue

        p_prob = (cnt_p + smoothing) / (n_prob + 2*smoothing)
        p_non  = (cnt_n + smoothing) / (n_non + 2*smoothing)

        rr = p_prob / p_non

        rows.append({
            "ingredient": ing,
            "problem_review_count": cnt_p,
            "non_problem_review_count": cnt_n,
            "problem_rate": p_prob,
            "non_problem_rate": p_non,
            "risk_ratio": rr
        })

    out = pd.DataFrame(rows).sort_values("risk_ratio", ascending=False)
    return out

risk_B = compute_risk_ratio("problem_B", min_problem_count=20, min_total_count=40, smoothing=0.5)
print("\n=== TOP risk ingredients (Definition B: symptom + rating<=2) ===")
print(risk_B.head(20).to_string(index=False))

risk_A = compute_risk_ratio("problem_A", min_problem_count=50, min_total_count=80, smoothing=0.5)
print("\n=== TOP risk ingredients (Definition A: symptom only) ===")
print(risk_A.head(20).to_string(index=False))


risk_B.head(50).to_csv("strategy2_riskratio_B_top50.csv", index=False)
risk_A.head(50).to_csv("strategy2_riskratio_A_top50.csv", index=False)
print("\nSaved: strategy2_riskratio_B_top50.csv and strategy2_riskratio_A_top50.csv")



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["figure.dpi"] = 150


plot_df = pd.DataFrame({
    "ingredient": [
        "mandelic acid",
        "zinc pca",
        "azelaic acid",
        "niacinamide",
        "tartaric acid",
        "potassium citrate",
        "salicylic acid"
    ],
    "risk_ratio": [
        2.530142,
        2.239107,
        2.227332,
        2.067871,
        2.066244,
        2.066244,
        1.825547
    ],
    "problem_review_count": [
        297,
        3807,
        629,
        3919,
        2073,
        2073,
        2765
    ]
})


plot_df = plot_df.sort_values("risk_ratio", ascending=True)


norm = plt.Normalize(plot_df["risk_ratio"].min(), plot_df["risk_ratio"].max())


colors = cm.Blues(0.3 + 0.7 * norm(plot_df["risk_ratio"]))


fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.barh(
    plot_df["ingredient"],
    plot_df["risk_ratio"],
    color=colors
)

ax.set_xlabel("Risk Ratio")
ax.set_ylabel("Active Ingredient")
ax.set_title(
    "Figure 3. Top Active Ingredients Associated with Symptom-Related Reviews\n"
    "(Risk Ratio Analysis After Removing Base Ingredients)"
)

ax.set_xlim(0, max(plot_df["risk_ratio"]) + 0.5)

# Add labels
for bar, rr, cnt in zip(bars, plot_df["risk_ratio"], plot_df["problem_review_count"]):
    ax.text(
        bar.get_width() + 0.03,
        bar.get_y() + bar.get_height() / 2,
        f"RR={rr:.2f} | n={cnt}",
        va="center"
    )

plt.tight_layout()
plt.savefig("strategy2_final_risk_ratio_chart_colored.png", bbox_inches="tight")
plt.show()

print("Saved file: strategy2_final_risk_ratio_chart_colored.png")