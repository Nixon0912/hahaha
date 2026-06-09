"""
Broker cost configuration — The5ers (FivePercentOnline-Real)
Account: 26395963 | Account currency: USD

Values taken directly from MT5 Symbol Specification windows.
Used by backtests / strategy code to model REAL trading costs.

Swap convention here: "In points". Money cost per lot per night =
    swap_points * contract_size * point
For XAUUSD: point = 0.01, contract_size = 100  ->  1 point = $1 per lot.
Triple-swap day multiplies the nightly swap by 3.

Commission convention: "% of notional per lot, charged in AND out".
    cost_per_side = notional * (commission_pct / 100)
    round_turn    = cost_per_side * 2
"""

BROKER = "FivePercentOnline-Real"
ACCOUNT_CURRENCY = "USD"

SYMBOLS = {
    "XAUUSD": {
        "description":      "Gold vs US Dollar",
        "sector":           "Commodities",
        "digits":           2,
        "point":            0.01,
        "contract_size":    100,        # ounces per 1.0 lot
        "tick_size":        0.01,
        "tick_value":       0.01,       # per 0.01 lot (min volume)
        "calculation":      "CFD Leverage",
        "margin_currency":  "USD",
        "profit_currency":  "USD",

        # Costs
        "spread_type":      "floating",
        "stops_level":      10,         # min distance (points) for SL/TP
        "swap_type":        "points",
        "swap_long":        -148.5,     # points per lot per night
        "swap_short":       -148.5,     # points per lot per night
        "triple_swap_day":  "Friday",   # nightly swap x3
        "commission_pct":   0.001,      # % of notional, per lot, per side
        "commission_in_out": True,      # charged on both entry and exit

        # Volume
        "volume_min":       0.01,
        "volume_max":       100,
        "volume_step":      0.01,

        # Trading hours (server time, GMT+? — confirm server offset)
        "session":          "01:05-23:50",  # Mon-Fri
    },
    # Add more symbols here as you send their specs:
    # "NAS100": {...},
    # "US30":   {...},
    # "EURUSD": {...},
}


# ── Cost helper functions ─────────────────────────────────────────────────────

def swap_cost_per_night(symbol: str, direction: str, lots: float, day: str = None) -> float:
    """
    Money cost (USD) of holding a position overnight.
    Negative = you pay. direction: 'long' or 'short'. day: e.g. 'Friday' for triple.
    """
    s = SYMBOLS[symbol]
    pts = s["swap_long"] if direction == "long" else s["swap_short"]
    money = pts * s["contract_size"] * s["point"] * lots
    if day and day == s["triple_swap_day"]:
        money *= 3
    return money


def commission_round_turn(symbol: str, price: float, lots: float) -> float:
    """Total commission (USD) for opening AND closing a position."""
    s = SYMBOLS[symbol]
    notional = price * s["contract_size"] * lots
    per_side = notional * (s["commission_pct"] / 100)
    return per_side * (2 if s["commission_in_out"] else 1)


def point_value(symbol: str, lots: float = 1.0) -> float:
    """USD value of a 1-point price move for the given lot size."""
    s = SYMBOLS[symbol]
    return s["contract_size"] * s["point"] * lots


if __name__ == "__main__":
    # Quick sanity check for XAUUSD at ~$3,300, 1 lot
    price, lots = 3300.0, 1.0
    print("XAUUSD cost model @ $3,300, 1.0 lot")
    print(f"  1-point value:        ${point_value('XAUUSD', lots):.2f}")
    print(f"  Commission round-turn: ${commission_round_turn('XAUUSD', price, lots):.2f}")
    print(f"  Swap long / night:    ${swap_cost_per_night('XAUUSD', 'long', lots):.2f}")
    print(f"  Swap long / Friday:   ${swap_cost_per_night('XAUUSD', 'long', lots, 'Friday'):.2f}")
