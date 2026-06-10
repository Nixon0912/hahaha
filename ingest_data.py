"""
Universal data ingest for XAUUSD history.

Drop any MetaTrader export into data/raw/ and run:  python3 ingest_data.py
It auto-detects format, normalises to our canonical schema, and (if you give
an intraday file) resamples a clean H1 for the trend/ATR indicators.

Accepted input formats (auto-detected):
  1. MT5 GUI export  : tab-separated, headers <DATE> <TIME> <OPEN> ... <SPREAD>
                       date "YYYY.MM.DD", time "HH:MM:SS"
  2. MT5 Python API  : comma CSV, columns datetime,open,high,low,close,tick_vol,spread
  3. Generic OHLC    : any CSV with a parseable timestamp col + o/h/l/c
                       (spread optional; if missing, a constant estimate is used)

Canonical output (what the backtester loads):
  data/XAUUSD_<TF>.csv  with columns: datetime,open,high,low,close,tick_vol,spread

Naming: put the timeframe in the filename (M5, M15, H1, etc.) OR pass it in the
mapping below. Files are matched case-insensitively.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, glob, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, "data", "raw")
OUT  = os.path.join(HERE, "data")
DEFAULT_SPREAD = 30   # points, used only if input has no spread column

def detect_tf(fname):
    m = re.search(r"\b(M1|M5|M10|M15|M20|M25|M30|H1|H4|D1|W1)\b", fname, re.I)
    return m.group(1).upper() if m else None

def load_any(path):
    head = pd.read_csv(path, sep=None, engine="python", nrows=0).columns.tolist()
    head_l = [str(c).lower().strip("<> ") for c in head]

    if "<DATE>" in head or "date" in head_l and "time" in head_l:
        # MT5 GUI tab-export
        sep = "\t" if "<DATE>" in head else None
        df = pd.read_csv(path, sep=sep, engine="python")
        df.columns = [c.strip("<> ").lower() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                        format="%Y.%m.%d %H:%M:%S", errors="coerce")
        df = df.rename(columns={"tickvol": "tick_vol"})
    elif "datetime" in head_l:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # generic: first column assumed timestamp
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        ts = df.columns[0]
        df["datetime"] = pd.to_datetime(df[ts], errors="coerce")

    for need in ["open", "high", "low", "close"]:
        if need not in df.columns:
            raise ValueError(f"{path}: missing '{need}' column (have {list(df.columns)})")
    if "tick_vol" not in df.columns:
        df["tick_vol"] = df.get("tickvol", df.get("volume", 0))
    if "spread" not in df.columns:
        df["spread"] = DEFAULT_SPREAD
        print(f"    (no spread column — using constant {DEFAULT_SPREAD} pts)")

    df = df[["datetime", "open", "high", "low", "close", "tick_vol", "spread"]]
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

def resample_h1(df):
    agg = {"open":"first","high":"max","low":"min","close":"last",
           "tick_vol":"sum","spread":"mean"}
    return df.resample("1h").agg(agg).dropna()

def main():
    os.makedirs(RAW, exist_ok=True)
    files = [f for f in glob.glob(os.path.join(RAW, "*"))
             if f.lower().endswith((".csv", ".txt"))]
    if not files:
        print(f"No files in {RAW}/")
        print("Drop your MT5 export(s) there (filename should contain the timeframe,")
        print("e.g. XAUUSD_M5_2019-2026.csv) and re-run.")
        return

    finest = None  # (tf_minutes, df) to build H1 from
    tf_min = {"M1":1,"M5":5,"M10":10,"M15":15,"M20":20,"M25":25,"M30":30,"H1":60,"H4":240,"D1":1440}
    for path in sorted(files):
        fn = os.path.basename(path)
        tf = detect_tf(fn)
        if tf is None:
            print(f"SKIP {fn}: cannot detect timeframe from filename")
            continue
        df = load_any(path)
        out = os.path.join(OUT, f"XAUUSD_{tf}.csv")
        df.to_csv(out)
        print(f"OK  {fn}  ->  XAUUSD_{tf}.csv   {len(df):,} bars  "
              f"{df.index[0]} -> {df.index[-1]}")
        if tf in tf_min and (finest is None or tf_min[tf] < finest[0]):
            finest = (tf_min[tf], df, tf)

    # Build H1 from the finest intraday file if no H1 supplied
    have_h1 = os.path.exists(os.path.join(OUT, "XAUUSD_H1.csv"))
    if finest and finest[0] < 60:
        h1 = resample_h1(finest[1])
        h1.to_csv(os.path.join(OUT, "XAUUSD_H1.csv"))
        src = f"resampled from {finest[2]}"
        print(f"H1  {src}   {len(h1):,} bars  {h1.index[0]} -> {h1.index[-1]}")
    elif not have_h1:
        print("WARN: no intraday file to build H1 from, and no H1 supplied.")

    print("\nDone. The backtester will now load the new data via load('<TF>').")

if __name__ == "__main__":
    main()
