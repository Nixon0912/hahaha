"""
Resample M5 XAUUSD data into higher timeframes.
Generates: M10, M15, M20, M25, M30, H1
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

TIMEFRAMES = {
    "M10": "10min",
    "M15": "15min",
    "M20": "20min",
    "M25": "25min",
    "M30": "30min",
    "H1":  "1h",
}


def load_mt5_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["date", "time"], inplace=True)
    df.rename(columns={"tickvol": "tick_vol", "vol": "real_vol"}, inplace=True)
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return pd.DataFrame({
        "open":     df["open"].resample(rule).first(),
        "high":     df["high"].resample(rule).max(),
        "low":      df["low"].resample(rule).min(),
        "close":    df["close"].resample(rule).last(),
        "tick_vol": df["tick_vol"].resample(rule).sum(),
        "spread":   df["spread"].resample(rule).mean().round(1),
    }).dropna(subset=["open"])


if __name__ == "__main__":
    print("Loading M5 base data...")
    m5 = load_mt5_csv("XAUUSD_M5.csv")
    print(f"  {len(m5):,} bars  ({m5.index.min()} → {m5.index.max()})\n")

    for tf_name, rule in TIMEFRAMES.items():
        out = resample(m5, rule)
        out_path = DATA_DIR / f"XAUUSD_{tf_name}.csv"
        out.to_csv(out_path)
        print(f"  XAUUSD_{tf_name}: {len(out):>6,} bars  → {out_path.name}")

    print("\nDone. All timeframes saved.")
