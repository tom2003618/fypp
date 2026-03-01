import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data = [
    {"Province": "Alberta", "Median age (July 1, 2025)": 38.1, "GDP (million CAD, 2024)": 473_937, "Population (July 1, 2024)": 4_909_030,
     "GDP per capita (CAD, 2024)": 96_544, "Market income per capita (CAD, 2023)": 44_916},
    {"Province": "British Columbia", "Median age (July 1, 2025)": 41.4, "GDP (million CAD, 2024)": 429_089, "Population (July 1, 2024)": 5_671_114,
     "GDP per capita (CAD, 2024)": 75_662, "Market income per capita (CAD, 2023)": 44_337},
    {"Province": "Ontario", "Median age (July 1, 2025)": 39.9, "GDP (million CAD, 2024)": 1_197_020, "Population (July 1, 2024)": 16_144_797,
     "GDP per capita (CAD, 2024)": 74_143, "Market income per capita (CAD, 2023)": 42_740},
    {"Province": "Quebec", "Median age (July 1, 2025)": 42.8, "GDP (million CAD, 2024)": 616_771, "Population (July 1, 2024)": 8_995_474,
     "GDP per capita (CAD, 2024)": 68_565, "Market income per capita (CAD, 2023)": 39_294},
]

df = pd.DataFrame(data).set_index("Province")
df_sorted_gdppc = df.sort_values("GDP per capita (CAD, 2024)", ascending=False)

# Save a comparison table as CSV too (optional)
csv_path = Path("/mnt/data/AB_BC_ON_QC_age_econ_comparison.csv")
df.to_csv(csv_path, encoding="utf-8")

# Chart 1: GDP per capita
plt.figure()
plt.bar(df_sorted_gdppc.index, df_sorted_gdppc["GDP per capita (CAD, 2024)"])
plt.ylabel("CAD per person (2024)")
plt.title("GDP per capita — AB, BC, ON, QC (2024)")
plt.xticks(rotation=20, ha="right")
png1 = Path("/mnt/data/gdp_per_capita_AB_BC_ON_QC_2024.png")
plt.tight_layout()
plt.savefig(png1, dpi=200)
plt.close()

# Chart 2: Median age
df_sorted_age = df.sort_values("Median age (July 1, 2025)", ascending=False)
plt.figure()
plt.bar(df_sorted_age.index, df_sorted_age["Median age (July 1, 2025)"])
plt.ylabel("Years")
plt.title("Median age — AB, BC, ON, QC (July 1, 2025)")
plt.xticks(rotation=20, ha="right")
png2 = Path("/mnt/data/median_age_AB_BC_ON_QC_2025.png")
plt.tight_layout()
plt.savefig(png2, dpi=200)
plt.close()

# Display table to user
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("AB/BC/ON/QC age & economic comparison", df.reset_index())
csv_path, png1, png2

