//+------------------------------------------------------------------+
//|  APEX9_EA.mq5                                                    |
//|  Signal-file executor for APEX-9 Python signal engine.          |
//|                                                                  |
//|  This EA does ONE job: poll signals.json and execute orders.     |
//|  All intelligence (ML, features, risk guards) lives in Python.   |
//|                                                                  |
//|  Setup:                                                          |
//|    1. Copy signals.json path into SIGNAL_FILE below             |
//|    2. Attach to ANY chart (e.g. EURUSD M1) — chart doesn't matter|
//|    3. Start Python runner: python -m ea.runner --signal-file-mode|
//+------------------------------------------------------------------+
#property copyright "APEX-9"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <JAson.mqh>   // Download from MQL5 community if not present

//--- inputs
// On Mac, MT5 reads/writes files in:
//   ~/Library/Application Support/MetaTrader 5/MQL5/Files/
// FILE_COMMON flag maps to that directory — just use the filename, no path needed.
input string SIGNAL_FILENAME = "apex9_signals.json"; // filename only, no path
input int    MAGIC_NUMBER    = 20260101;
input int    DEVIATION_PTS   = 20;
input bool   VERBOSE         = true;

CTrade trade;
datetime lastFileTime = 0;

//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber(MAGIC_NUMBER);
    trade.SetDeviationInPoints(DEVIATION_PTS);
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    EventSetTimer(10); // check every 10 seconds
    Print("APEX9 EA initialized. Watching: ", SIGNAL_FILENAME);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTimer()
{
    ProcessSignalFile();
    CheckForceClose();
}

//+------------------------------------------------------------------+
void OnTick() { } // not used

//+------------------------------------------------------------------+
void ProcessSignalFile()
{
    // FILE_COMMON maps to ~/Library/Application Support/MetaTrader 5/MQL5/Files/ on Mac
    if(!FileIsExist(SIGNAL_FILENAME, FILE_COMMON)) return;

    int fh = FileOpen(SIGNAL_FILENAME, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
    if(fh == INVALID_HANDLE) return;

    string content = "";
    while(!FileIsEnding(fh))
        content += FileReadString(fh);
    FileClose(fh);

    if(StringLen(content) < 5) return;

    // Parse JSON array
    CJAVal json;
    if(!json.Deserialize(content)) { Print("JSON parse error"); return; }

    bool anyPending = false;

    for(int i = 0; i < json.Size(); i++)
    {
        string status = json[i]["status"].ToStr();

        if(status == "pending")
        {
            anyPending = true;
            string  sym  = json[i]["symbol"].ToStr();
            int     dir  = (int)json[i]["direction"].ToInt();  // 1=buy, -1=sell
            double  lots = json[i]["lots"].ToDbl();
            double  sl   = json[i]["sl"].ToDbl();
            double  tp   = json[i]["tp"].ToDbl();
            string  cmt  = json[i]["comment"].ToStr();

            bool ok = ExecuteOrder(sym, dir, lots, sl, tp, cmt);
            json[i]["status"] = ok ? "executed" : "failed";
            json[i]["executed_at"] = TimeToString(TimeCurrent());
        }
        else if(status == "close_all")
        {
            string sym = json[i]["symbol"].ToStr();
            CloseAllForSymbol(sym);
            json[i]["status"] = "closed";
        }
    }

    if(anyPending)
    {
        // Write back updated statuses (FILE_COMMON keeps it in the same sandbox dir)
        int fw = FileOpen(SIGNAL_FILENAME, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
        if(fw != INVALID_HANDLE)
        {
            FileWriteString(fw, json.Serialize());
            FileClose(fw);
        }
    }
}

//+------------------------------------------------------------------+
bool ExecuteOrder(string sym, int dir, double lots, double sl, double tp, string cmt)
{
    // Verify symbol exists
    if(!SymbolSelect(sym, true)) { Print("Symbol not found: ", sym); return false; }

    double price = (dir == 1) ? SymbolInfoDouble(sym, SYMBOL_ASK)
                               : SymbolInfoDouble(sym, SYMBOL_BID);
    int    digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
    sl = NormalizeDouble(sl, digits);
    tp = NormalizeDouble(tp, digits);

    bool ok;
    if(dir == 1)
        ok = trade.Buy(lots, sym, price, sl, tp, cmt);
    else
        ok = trade.Sell(lots, sym, price, sl, tp, cmt);

    if(ok)
        Print("ORDER PLACED: ", sym, " ", (dir==1?"BUY":"SELL"),
              " ", lots, " SL=", sl, " TP=", tp);
    else
        Print("ORDER FAILED: ", sym, " retcode=", trade.ResultRetcode(),
              " ", trade.ResultRetcodeDescription());
    return ok;
}

//+------------------------------------------------------------------+
void CloseAllForSymbol(string sym)
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL) != sym) continue;
        if(PositionGetInteger(POSITION_MAGIC) != MAGIC_NUMBER) continue;
        trade.PositionClose(ticket);
        Print("Closed: ", sym, " ticket=", ticket);
    }
}

//+------------------------------------------------------------------+
void CheckForceClose()
{
    // Redundant force-close: 20:55, 20:57, 20:59, 21:00 server time
    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    bool inWindow = (dt.hour == 20 && (dt.min == 55 || dt.min == 57 || dt.min == 59))
                 || (dt.hour == 21 && dt.min == 0);
    if(!inWindow) return;

    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetInteger(POSITION_MAGIC) != MAGIC_NUMBER) continue;
        trade.PositionClose(ticket);
        count++;
    }
    if(count > 0)
        Print("FORCE CLOSE (", dt.hour, ":", dt.min, "): closed ", count, " positions");
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    EventKillTimer();
    // Emergency close on EA removal
    CheckForceClose();
}
