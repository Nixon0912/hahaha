"""
Load MT5-exported CSV data into a clean pandas DataFrame.

MT5 export format (tab-separated):
    <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_mt5_csv(filename: str) -> pd.DataFrame:
    """Load an MT5-exported CSV into a clean OHLCV DataFrame indexed by datetime."""
    path = DATA_DIR / filename
    df = pd.read_csv(path, sep="\t")

    # Strip the angle brackets from MT5 column names: <OPEN> -> open
    df.columns = [c.strip("<>").lower() for c in df.columns]

    # Combine date + time into a single datetime index
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    format="%Y.%m.%d %H:%M:%S")
    df.set_index("datetime", inplace=True)
    df.drop(columns=["date", "time"], inplace=True)

    # Rename to friendly names
    df.rename(columns={"tickvol": "tick_vol", "vol": "real_vol"}, inplace=True)
    return df


if __name__ == "__main__":
    df = load_mt5_csv("XAUUSD_M5.csv")
    print(f"Rows:        {len(df):,}")
    print(f"Date range:  {df.index.min()}  ->  {df.index.max()}")
    print(f"Columns:     {list(df.columns)}")
    print(f"\nSpread (broker points)  min/mean/max: "
          f"{df['spread'].min()} / {df['spread'].mean():.1f} / {df['spread'].max()}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nLast 3 rows:")
    print(df.tail(3))

    # Basic integrity checks
    print("\n--- Integrity ---")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate timestamps: {df.index.duplicated().sum()}")
    bad = df[(df["high"] < df["low"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])]
    print(f"Bad OHLC bars: {len(bad)}")
