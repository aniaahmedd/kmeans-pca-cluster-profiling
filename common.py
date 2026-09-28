from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "penguins.csv"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
SEED = 42
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
def load_raw(): return pd.read_csv(DATA)
