"""
Per-instrument metadata for the multi-asset framework.

For a fixed-fractional-risk backtest we only need:
  tick    : price increment of 1 "point" in the data's spread column
  session : when the instrument's main breakout session is (local data time)

Everything else (P&L) is expressed in R-multiples (risk units), so exact
contract/lot value is not needed to find statistical edge.
"""

# tick = price value of 1 unit in the <SPREAD> column.
# Determined from the instrument's decimal convention.
TICK = {
    "XAUUSD": 0.01,   "XAGUSD": 0.001,
    "XPTUSD": 0.01,   "XPDUSD": 0.01,
    "ASXAUD": 0.01,
    "XTIUSD": 0.01,   "XBRUSD": 0.01,  "NGCUSD": 0.001, "CUCUSD": 0.0001,
    "EURUSD": 0.00001,"AUDUSD": 0.00001,"AUDNZD":0.00001,"USDCAD":0.00001,
    "USDCHF": 0.00001,"USDJPY": 0.001,
    "DAX40": 0.1, "F40EUR":0.1, "ESXEUR":0.1, "IBXEUR":0.1, "UK100":0.1,
    "NAS100":0.1, "SP500":0.1, "US30":1.0, "JPN225":1.0, "HSIHKD":1.0,
}

# Asset class grouping (for correlation-aware portfolio construction)
CLASS = {
    "XAUUSD":"metal","XAGUSD":"metal","XPTUSD":"metal","XPDUSD":"metal","ASXAUD":"metal",
    "XTIUSD":"energy","XBRUSD":"energy","NGCUSD":"energy","CUCUSD":"metal",
    "EURUSD":"fx","AUDUSD":"fx","AUDNZD":"fx","USDCAD":"fx","USDCHF":"fx","USDJPY":"fx",
    "DAX40":"index","F40EUR":"index","ESXEUR":"index","IBXEUR":"index","UK100":"index",
    "NAS100":"index","SP500":"index","US30":"index","JPN225":"index","HSIHKD":"index",
}

def tick_for(sym):
    return TICK.get(sym, 0.01)
