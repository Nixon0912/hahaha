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
input string SIGNAL_FILE  = "C:\\apex9\\ea\\signals.json"; // Path to signals.json
input int    MAGIC_NUMBER  = 20260101;
input int    DEVIATION_PTS = 20;
input bool   VERBOSE       = true;

CTrade trade;
datetime lastFileTime = 0;

//+------------------------------------------------------------------+
int OnInit()
{
    trade.SetExpertMagicNumber(MAGIC_NUMBER);
    trade.SetDeviationInPoints(DEVIATION_PTS);
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    EventSetTimer(10); // check every 10 seconds
    Print("APEX9 EA initialized. Watching: ", SIGNAL_FILE);
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
    if(!FileIsExist(SIGNAL_FILE)) return;

    // Read file
    int fh = FileOpen(SIGNAL_FILE, FILE_READ | FILE_TXT | FILE_ANSI);
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
    CJAVal updated;
    updated.IsArray(true);

    for(int i = 0; i < json.Size(); i++)
    {
        CJAVal sig = json[i];
        string status = sig["status"].ToStr();

        if(status == "pending")
        {
            anyPending = true;
            string  sym  = sig["symbol"].ToStr();
            int     dir  = (int)sig["direction"].ToInt();  // 1=buy, -1=sell
            double  lots = sig["lots"].ToDbl();
            double  sl   = sig["sl"].ToDbl();
            double  tp   = sig["tp"].ToDbl();
            string  cmt  = sig["comment"].ToStr();

            bool ok = ExecuteOrder(sym, dir, lots, sl, tp, cmt);
            sig["status"] = ok ? "executed" : "failed";
            sig["executed_at"] = TimeToString(TimeCurrent());
        }
        else if(status == "close_all")
        {
            string sym = sig["symbol"].ToStr();
            CloseAllForSymbol(sym);
            sig["status"] = "closed";
        }

        updated.Add(sig);
    }

    if(anyPending)
    {
        // Write back updated statuses
        int fw = FileOpen(SIGNAL_FILE, FILE_WRITE | FILE_TXT | FILE_ANSI);
        if(fw != INVALID_HANDLE)
        {
            FileWriteString(fw, updated.Serialize());
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
