
import csv
import io
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Trade Analyzer (IBKR + Fidelity)", layout="wide")

APP_PASS = os.getenv("APP_PASS", "")

if APP_PASS:
    code = st.sidebar.text_input("Access code", type="password")
    if code != APP_PASS:
        st.title("🔒 Trade Analyzer")
        if code:
            st.sidebar.error("Wrong access code.")
        else:
            st.sidebar.warning("Enter access code to continue.")
        st.stop()

# ---------------------------------------------------------------------------------------



DATE_LINE_RE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*,")
HEADER_RE = re.compile(r"^\s*Run Date\s*,", re.IGNORECASE)
POST_MORTEM_TAGS = [
    "Entered too early / chased",
    "Held too long",
    "Wrong thesis / news",
    "Position size too big",
    "Didn’t cut at stop",
    "IV crush / earnings",
    "Rolled to avoid loss",
]
UNASSIGNED_TAG = "(unassigned)"

def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """
    Tries to load CSV (comma or tab). If it looks like a Fidelity export with disclaimer footer,
    it strips non-table lines and retries.
    """
    text = file_bytes.decode("utf-8-sig", errors="replace")

    # Heuristic sep guess on first chunk
    sample = text[:4096]
    sep = "\t" if sample.count("\t") > sample.count(",") else ","

    def _read(t: str, sep_: str) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(t), sep=sep_, engine="python", skip_blank_lines=True)

    def _read_flexible(t: str, sep_: str) -> pd.DataFrame:
        rows = []
        reader = csv.reader(io.StringIO(t), delimiter=sep_)
        for row in reader:
            if any(cell.strip() for cell in row):
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        header = rows[0]
        width = len(header)
        fixed_rows = []
        for row in rows[1:]:
            if len(row) < width:
                row = row + [""] * (width - len(row))
            elif len(row) > width:
                row = row[: width - 1] + [sep_.join(row[width - 1 :])]
            fixed_rows.append(row)
        return pd.DataFrame(fixed_rows, columns=header)

    try:
        df = _read(text, sep)
        # If it parsed into 1 column and looks wrong, fall through to clean read
        if df.shape[1] == 1 and (str(df.columns[0]).lower().startswith("unnamed") or "," in str(df.columns[0])):
            raise ValueError("Single-column parse; attempting cleaned parse")
        return df
    except Exception:
        try:
            df = _read_flexible(text, sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
        # Clean: keep header + date-starting lines only
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # find header
        header_idx = None
        for i, ln in enumerate(lines):
            if HEADER_RE.match(ln):
                header_idx = i
                break
        if header_idx is None:
            # Last resort: return the original error by re-raising
            raise

        keep = [lines[header_idx]]
        for ln in lines[header_idx + 1:]:
            if DATE_LINE_RE.match(ln):
                keep.append(ln)
            else:
                # stop at first non-date line after table begins (Fidelity disclaimer/footer)
                break

        cleaned = "\n".join(keep)
        # Fidelity is comma-separated
        try:
            df = _read(cleaned, ",")
        except Exception:
            df = _read_flexible(cleaned, ",")
        return df

# -------------------------
# Option parsing
# -------------------------

HUMAN_OPT_RE = re.compile(r"(?P<under>[A-Z]{1,6})\s+(?P<exp>\d{1,2}/\d{1,2}/\d{2,4})\s+(?P<strike>\d+(?:\.\d+)?)\s+(?P<right>[CP])\b")
OCC_OPT_RE = re.compile(r"^(?P<under>[A-Z ]{1,6})(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})(?P<right>[CP])(?P<k>\d{8})$")
FID_OPT_RE = re.compile(r"^-?(?P<under>[A-Z]{1,6})(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})(?P<right>[CP])(?P<k>\d{1,6})(?:\.\d+)?$")
FID_DESC_RE = re.compile(r"\b(?P<mon>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+(?P<day>\d{1,2})\s+(?P<yy>\d{2})\s+\$(?P<strike>\d+(?:\.\d+)?)\b", re.IGNORECASE)
MON_MAP = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

def _parse_mmddyy(s: str) -> Optional[pd.Timestamp]:
    dt = pd.to_datetime(str(s).strip(), errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt.date())

def parse_occ(symbol_text: str) -> Optional[Tuple[str, pd.Timestamp, float, str]]:
    t = str(symbol_text).strip().replace(" ", "")
    m = OCC_OPT_RE.match(t)
    if not m:
        return None
    under = m.group("under").strip()
    year = 2000 + int(m.group("y"))
    exp = pd.Timestamp(datetime(year, int(m.group("m")), int(m.group("d"))).date())
    right = m.group("right")
    strike = int(m.group("k")) / 1000.0
    return under, exp, float(strike), right

def parse_fidelity_symbol(sym: str) -> Optional[Tuple[str, pd.Timestamp, float, str]]:
    t = str(sym).strip().upper()
    m = FID_OPT_RE.match(t)
    if not m:
        return None
    under = m.group("under")
    year = 2000 + int(m.group("y"))
    exp = pd.Timestamp(datetime(year, int(m.group("m")), int(m.group("d"))).date())
    right = m.group("right")
    strike = float(m.group("k"))
    return under, exp, strike, right

def parse_fidelity_description(desc: str) -> Optional[Tuple[pd.Timestamp, float]]:
    t = str(desc).upper()
    m = FID_DESC_RE.search(t)
    if not m:
        return None
    mon = MON_MAP.get(m.group("mon").upper())
    day = int(m.group("day"))
    year = 2000 + int(m.group("yy"))
    exp = pd.Timestamp(datetime(year, mon, day).date())
    strike = float(m.group("strike"))
    return exp, strike

def parse_option(underlying_hint: Optional[str], symbol_text: str, desc_text: Optional[str]) -> Optional[Tuple[str, pd.Timestamp, float, str]]:
    out = parse_occ(symbol_text)
    if out:
        return out
    out = parse_fidelity_symbol(symbol_text)
    if out:
        return out
    m = HUMAN_OPT_RE.search(str(symbol_text).upper())
    if m:
        exp = _parse_mmddyy(m.group("exp"))
        if exp is None:
            return None
        return m.group("under"), exp, float(m.group("strike")), m.group("right")
    if desc_text is not None:
        td = str(desc_text).upper()
        right = "C" if "CALL" in td else ("P" if "PUT" in td else None)
        if right and underlying_hint:
            exs = parse_fidelity_description(td)
            if exs:
                exp, strike = exs
                return str(underlying_hint).strip().upper(), exp, float(strike), right
    return None

# -------------------------
# FIFO matching
# -------------------------
@dataclass
class Lot:
    qty: int
    price: float
    fees: float
    date: pd.Timestamp
    side: str
    multiplier: float

def fifo_match(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    buys: List[Lot] = []
    sells: List[Lot] = []
    closed = []

    for _, r in trades.iterrows():
        lot = Lot(
            qty=int(r["qty"]),
            price=float(r["price"]),
            fees=float(r["fees"]),
            date=r["trade_dt"],
            side=r["side"],
            multiplier=float(r["multiplier"]),
        )
        if lot.side == "BUY":
            buys.append(lot)
        else:
            sells.append(lot)

        def match_once():
            nonlocal buys, sells, closed
            if not buys or not sells:
                return False
            b, s = buys[0], sells[0]
            mqty = min(b.qty, s.qty)
            multiplier = float(b.multiplier if b.multiplier is not None else s.multiplier)

            gross = (s.price - b.price) * mqty * multiplier
            b_fee = b.fees * (mqty / b.qty) if b.qty else 0.0
            s_fee = s.fees * (mqty / s.qty) if s.qty else 0.0
            net = gross - (b_fee + s_fee)

            closed.append({
                "buy_date": b.date,
                "sell_date": s.date,
                "close_date": max(b.date, s.date),
                "qty": mqty,
                "buy_price": b.price,
                "sell_price": s.price,
                "gross_pnl": gross,
                "fees": (b_fee + s_fee),
                "net_pnl": net
            })

            b.qty -= mqty; s.qty -= mqty
            b.fees -= b_fee; s.fees -= s_fee
            if b.qty == 0: buys.pop(0)
            if s.qty == 0: sells.pop(0)
            return True

        while match_once():
            pass

    open_rows = []
    for lot in buys:
        open_rows.append({
            "side": "BUY",
            "qty": lot.qty,
            "price": lot.price,
            "fees": lot.fees,
            "date": lot.date,
            "multiplier": lot.multiplier,
        })
    for lot in sells:
        open_rows.append({
            "side": "SELL",
            "qty": lot.qty,
            "price": lot.price,
            "fees": lot.fees,
            "date": lot.date,
            "multiplier": lot.multiplier,
        })

    return pd.DataFrame(closed), pd.DataFrame(open_rows)

# -------------------------
# Mistake detector helpers
# -------------------------
def _contract_key(df: pd.DataFrame) -> pd.Series:
    expiry = pd.to_datetime(df["expiry"], errors="coerce").dt.date.astype(str)
    strike = pd.to_numeric(df["strike"], errors="coerce").astype(str)
    return df["underlying"].astype(str) + "|" + expiry + "|" + strike + "|" + df["right"].astype(str)

def detect_mistakes(closed_trades: pd.DataFrame, fills: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if closed_trades.empty:
        empty = pd.DataFrame(columns=[
            "mistake_type", "severity", "close_date", "underlying", "contract_key",
            "trade_id", "details", "net_pnl"
        ])
        return {
            "mistakes_df": empty,
            "summary_by_type": empty,
            "summary_by_week": empty,
            "examples_by_type": empty,
            "held_tables": {},
        }

    base = closed_trades.copy()
    base["close_date"] = pd.to_datetime(base["close_date"], errors="coerce")
    base["open_date"] = pd.to_datetime(base["open_date"], errors="coerce")
    base["trade_date"] = base["close_date"].dt.date
    base["holding_days"] = (base["close_date"] - base["open_date"]).dt.total_seconds() / (24 * 3600)
    base["contract_key"] = _contract_key(base)
    base["net_pnl"] = pd.to_numeric(base["net_pnl"], errors="coerce").fillna(0.0)

    mistakes: List[Dict[str, object]] = []

    # Rule 1: Overtrading day
    day_stats = (base.groupby("trade_date", as_index=False)
                 .agg(day_trade_count=("net_pnl", "size"), day_pnl=("net_pnl", "sum")))
    if not day_stats.empty:
        threshold = float(np.percentile(day_stats["day_trade_count"], 80))
        over = day_stats[(day_stats["day_trade_count"] >= threshold) & (day_stats["day_pnl"] < 0)]
        for _, row in over.iterrows():
            mistakes.append({
                "mistake_type": "Overtrading day",
                "severity": abs(float(row["day_pnl"])),
                "close_date": pd.Timestamp(row["trade_date"]),
                "underlying": "MULTI",
                "contract_key": "MULTI",
                "trade_id": str(row["trade_date"]),
                "details": f"{int(row['day_trade_count'])} closes, day PnL {row['day_pnl']:.2f}",
                "net_pnl": float(row["day_pnl"]),
            })

    # Rule 2: Big loss outlier
    losers = base[base["net_pnl"] < 0]
    winners = base[base["net_pnl"] > 0]
    if not losers.empty:
        base_loss = float(np.median(np.abs(losers["net_pnl"])))
        avg_win = float(winners["net_pnl"].mean()) if not winners.empty else 0.0
        threshold = max(3 * base_loss, 2 * avg_win) if avg_win > 0 else 3 * base_loss
        flagged = losers[np.abs(losers["net_pnl"]) >= threshold]
        for _, row in flagged.iterrows():
            mistakes.append({
                "mistake_type": "Big loss outlier",
                "severity": abs(float(row["net_pnl"])),
                "close_date": row["close_date"],
                "underlying": row["underlying"],
                "contract_key": row["contract_key"],
                "trade_id": str(row.get("trade_id", row.name)),
                "details": f"Loss {row['net_pnl']:.2f} exceeds threshold",
                "net_pnl": float(row["net_pnl"]),
            })

    # Rule 3: Held losers longer than winners
    held_tables = {}
    hold_ok = base["open_date"].notna().all()
    if hold_ok:
        winners_hold = base[base["net_pnl"] > 0]["holding_days"].dropna()
        losers_hold = base[base["net_pnl"] < 0]["holding_days"].dropna()
        if len(winners_hold) >= 5 and len(losers_hold) >= 5:
            med_w = float(np.median(winners_hold))
            med_l = float(np.median(losers_hold))
            if med_l > med_w * 1.25:
                last_close = base["close_date"].max()
                mistakes.append({
                    "mistake_type": "Held losers longer",
                    "severity": med_l - med_w,
                    "close_date": last_close,
                    "underlying": "MULTI",
                    "contract_key": "MULTI",
                    "trade_id": "global",
                    "details": f"Median hold losers {med_l:.1f}d vs winners {med_w:.1f}d",
                    "net_pnl": np.nan,
                })

        held_tables = {
            "losers": base[base["net_pnl"] < 0].sort_values("holding_days", ascending=False).head(10),
            "winners": base[base["net_pnl"] > 0].sort_values("holding_days", ascending=False).head(10),
        }

    # Rule 4: Gave back gains in ticker
    ticker = (base.groupby("underlying", as_index=False)
              .agg(ticker_net=("net_pnl", "sum"),
                   total_trades=("net_pnl", "size"),
                   has_wins=("net_pnl", lambda s: bool((s > 0).any())),
                   has_losses=("net_pnl", lambda s: bool((s < 0).any())),
                   last_close=("close_date", "max")))
    flagged = ticker[(ticker["ticker_net"] < 0) & ticker["has_wins"] & ticker["has_losses"] & (ticker["total_trades"] >= 5)]
    for _, row in flagged.iterrows():
        mistakes.append({
            "mistake_type": "Gave back gains in ticker",
            "severity": abs(float(row["ticker_net"])),
            "close_date": row["last_close"],
            "underlying": row["underlying"],
            "contract_key": "MULTI",
            "trade_id": str(row["underlying"]),
            "details": f"Net {row['ticker_net']:.2f} across {int(row['total_trades'])} trades with both wins & losses",
            "net_pnl": float(row["ticker_net"]),
        })

    # Rule 5: Averaged down losers (fills-based)
    if fills is not None and not fills.empty:
        fills_work = fills.copy()
        fills_work["contract_key"] = _contract_key(fills_work)
        for contract_key, g in fills_work.sort_values("trade_dt").groupby("contract_key", sort=False):
            net_pos = 0
            add_events = 0
            for _, r in g.iterrows():
                side = str(r["side"]).upper()
                qty = int(r["qty"])
                before = net_pos
                if side == "BUY":
                    net_pos += qty
                    if before >= 0 and net_pos > before:
                        add_events += 1
                elif side == "SELL":
                    net_pos -= qty
                    if before <= 0 and net_pos < before:
                        add_events += 1
            contract_net = float(base.loc[base["contract_key"] == contract_key, "net_pnl"].sum())
            if add_events >= 2 and contract_net < 0:
                parts = contract_key.split("|")
                mistakes.append({
                    "mistake_type": "Averaged down (added to loser)",
                    "severity": abs(contract_net),
                    "close_date": base.loc[base["contract_key"] == contract_key, "close_date"].max(),
                    "underlying": parts[0] if parts else "UNKNOWN",
                    "contract_key": contract_key,
                    "trade_id": contract_key,
                    "details": f"{add_events} adds, net {contract_net:.2f}",
                    "net_pnl": contract_net,
                })

    mistakes_df = pd.DataFrame(mistakes)
    if mistakes_df.empty:
        summary_by_type = pd.DataFrame(columns=["mistake_type", "count", "total_damage", "avg_damage"])
        summary_by_week = pd.DataFrame(columns=["week_key", "mistake_type", "count", "total_damage"])
        examples_by_type = pd.DataFrame(columns=mistakes_df.columns)
    else:
        mistakes_df["severity"] = pd.to_numeric(mistakes_df["severity"], errors="coerce").fillna(0.0)
        mistakes_df["close_date"] = pd.to_datetime(mistakes_df["close_date"], errors="coerce")
        summary_by_type = (mistakes_df.groupby("mistake_type", as_index=False)
                           .agg(count=("mistake_type", "size"),
                                total_damage=("severity", "sum"),
                                avg_damage=("severity", "mean"))
                           .sort_values("total_damage", ascending=False))
        iso = mistakes_df["close_date"].dt.isocalendar()
        week_key = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
        summary_by_week = (mistakes_df.assign(week_key=week_key)
                           .groupby(["week_key", "mistake_type"], as_index=False)
                           .agg(count=("mistake_type", "size"),
                                total_damage=("severity", "sum")))
        examples_by_type = mistakes_df.sort_values("severity", ascending=False)

    return {
        "mistakes_df": mistakes_df,
        "summary_by_type": summary_by_type,
        "summary_by_week": summary_by_week,
        "examples_by_type": examples_by_type,
        "held_tables": held_tables,
    }

# -------------------------
# Broker detection + normalization
# -------------------------
def detect_broker(columns: List[str]) -> str:
    cols = {str(c).strip().lower() for c in columns}
    if {"run date", "action", "symbol", "quantity", "price"}.issubset(cols):
        return "Fidelity"
    if any(str(c).strip().lower() in {"date/time", "date", "timestamp", "trade date"} for c in columns) and ("symbol" in cols or "description" in cols):
        return "IBKR"
    return "Unknown"

def _colmap(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).strip().lower(): c for c in df.columns}

def _series(df: pd.DataFrame, name: str, default="") -> pd.Series:
    cmap = _colmap(df)
    key = str(name).strip().lower()
    col = cmap.get(key)
    if col is None:
        for k, v in cmap.items():
            if key in k or k in key:
                col = v
                break
    if col is None:
        return pd.Series([default] * len(df), index=df.index)
    return df[col]

def normalize_fidelity(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()

    run_date = _series(w, "Run Date", default=np.nan)
    w["trade_dt"] = pd.to_datetime(run_date, errors="coerce")
    w = w.dropna(subset=["trade_dt"])

    action = _series(w, "Action", default="").astype(str).str.upper()
    qty_raw = pd.to_numeric(_series(w, "Quantity", default=0.0), errors="coerce").fillna(0.0)

    w["side"] = np.where(action.str.contains("SOLD"), "SELL",
                 np.where(action.str.contains("BOUGHT"), "BUY",
                 np.where(qty_raw < 0, "SELL", "BUY")))
    w["qty"] = qty_raw.abs().astype(int)

    w["price"] = pd.to_numeric(_series(w, "Price", default=np.nan), errors="coerce")
    w = w.dropna(subset=["price"])

    comm = pd.to_numeric(_series(w, "Commission", default=0.0), errors="coerce").fillna(0.0)
    fees = pd.to_numeric(_series(w, "Fees", default=0.0), errors="coerce").fillna(0.0)
    w["fees"] = (comm + fees).astype(float)

    w["multiplier"] = 100.0

    sym = _series(w, "Symbol", default="").astype(str)
    desc = _series(w, "Description", default="").astype(str)
    underlying_hint = sym.str.replace("^-", "", regex=True).str.extract(r"^([A-Z]{1,6})", expand=False)

    parsed = [parse_option(uh, s, d) for uh, s, d in zip(underlying_hint, sym, desc)]
    ok = [p is not None for p in parsed]
    w = w[ok].copy()
    w[["underlying", "expiry", "strike", "right"]] = pd.DataFrame([p for p in parsed if p is not None], index=w.index)

    w["contract_id"] = (w["underlying"].astype(str) + "|" + w["expiry"].astype(str) + "|" +
                        w["strike"].astype(str) + "|" + w["right"].astype(str))
    return w.sort_values("trade_dt").reset_index(drop=True)

def normalize_ibkr(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    w = df.copy()

    w["trade_dt"] = pd.to_datetime(w[mapping["datetime"]], errors="coerce")
    w = w.dropna(subset=["trade_dt"])

    side_raw = w[mapping["side"]].astype(str).str.upper().str.strip()
    w["side"] = np.where(side_raw.str.contains("SELL"), "SELL", np.where(side_raw.str.contains("BUY"), "BUY", side_raw))

    w["qty"] = pd.to_numeric(w[mapping["qty"]], errors="coerce").fillna(0).astype(int).abs()
    w["price"] = pd.to_numeric(w[mapping["price"]], errors="coerce")
    w = w.dropna(subset=["price"])

    feescol = mapping.get("fees")
    w["fees"] = pd.to_numeric(w[feescol], errors="coerce").fillna(0.0).astype(float) if feescol else 0.0

    multcol = mapping.get("multiplier")
    w["multiplier"] = pd.to_numeric(w[multcol], errors="coerce").fillna(100.0).astype(float) if multcol else 100.0

    sym = w[mapping["symbol"]].astype(str)
    parsed = [parse_option(None, s, None) for s in sym]
    ok = [p is not None for p in parsed]
    w = w[ok].copy()
    w[["underlying", "expiry", "strike", "right"]] = pd.DataFrame([p for p in parsed if p is not None], index=w.index)

    w["contract_id"] = (w["underlying"].astype(str) + "|" + w["expiry"].astype(str) + "|" +
                        w["strike"].astype(str) + "|" + w["right"].astype(str))
    return w.sort_values("trade_dt").reset_index(drop=True)

# -------------------------
# UI
# -------------------------
st.title("Trade Analyzer (IBKR + Fidelity) — Options FIFO P&L → Excel")
st.caption("Supports Fidelity 'Accounts History' CSV (with disclaimer footer) and IBKR CSV (with mapping).")

uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])
if not uploaded:
    st.stop()

try:
    df = load_csv_bytes(uploaded.getvalue())
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

detected = detect_broker(list(df.columns))
choice = st.selectbox("Broker format", ["Auto", "Fidelity", "IBKR"], index=0)
broker = detected if choice == "Auto" else choice
st.info(f"Detected: **{detected}** | Using: **{broker}**")

with st.expander("Debug: columns"):
    st.write(list(df.columns))
    st.write(df.head(3))

work = None

if broker == "Fidelity":
    try:
        work = normalize_fidelity(df)
    except Exception as e:
        st.error(f"Failed to parse Fidelity file: {e}")
        st.stop()
elif broker == "IBKR":
    cols = list(df.columns)

    def pick_like(candidates):
        for cand in candidates:
            for c in cols:
                if cand.lower() == str(c).lower():
                    return c
        for cand in candidates:
            for c in cols:
                if cand.lower() in str(c).lower():
                    return c
        return cols[0] if cols else None

    st.subheader("IBKR column mapping")
    c1, c2 = st.columns(2)
    with c1:
        col_datetime = st.selectbox("Trade datetime column", cols, index=cols.index(pick_like(["Date/Time", "Date", "TradeDate", "Timestamp"])))
        col_symbol = st.selectbox("Symbol / Description column", cols, index=cols.index(pick_like(["Symbol", "Description", "Instrument"])))
        col_side = st.selectbox("Side column (BUY/SELL)", cols, index=cols.index(pick_like(["Side", "Action", "Buy/Sell", "B/S"])))
        col_qty = st.selectbox("Quantity column", cols, index=cols.index(pick_like(["Quantity", "Qty", "Size"])))
    with c2:
        col_price = st.selectbox("Price column (per contract premium)", cols, index=cols.index(pick_like(["Price", "TradePrice", "FillPrice"])))
        col_fees = st.selectbox("Fees/Commission column (optional)", ["(none)"] + cols, index=0)
        col_mult = st.selectbox("Multiplier column (optional)", ["(none)"] + cols, index=0)

    mapping = {
        "datetime": col_datetime,
        "symbol": col_symbol,
        "side": col_side,
        "qty": col_qty,
        "price": col_price,
        "fees": None if col_fees == "(none)" else col_fees,
        "multiplier": None if col_mult == "(none)" else col_mult,
    }

    try:
        work = normalize_ibkr(df, mapping)
    except Exception as e:
        st.error(f"Failed to parse IBKR file: {e}")
        st.stop()
else:
    st.error("Could not auto-detect broker. Choose Fidelity or IBKR.")
    st.stop()

st.subheader("Parsed options preview")
if work is None or work.empty:
    st.warning("No option trades parsed from this file.")
    st.stop()

st.dataframe(work[["trade_dt", "underlying", "expiry", "strike", "right", "side", "qty", "price", "fees"]].head(80), use_container_width=True)

# FIFO per contract
closed_all = []
open_all = []
for cid, g in work.groupby("contract_id", sort=False):
    g2 = g[["trade_dt", "side", "qty", "price", "fees", "multiplier", "underlying", "expiry", "strike", "right"]].sort_values("trade_dt")
    closed_df, open_df = fifo_match(g2)
    if not closed_df.empty:
        closed_df["contract_id"] = cid
        closed_df["underlying"] = g2["underlying"].iloc[0]
        closed_df["expiry"] = g2["expiry"].iloc[0]
        closed_df["strike"] = g2["strike"].iloc[0]
        closed_df["right"] = g2["right"].iloc[0]
        closed_all.append(closed_df)
    if not open_df.empty:
        open_df["contract_id"] = cid
        open_df["underlying"] = g2["underlying"].iloc[0]
        open_df["expiry"] = g2["expiry"].iloc[0]
        open_df["strike"] = g2["strike"].iloc[0]
        open_df["right"] = g2["right"].iloc[0]
        open_all.append(open_df)

closed = pd.concat(closed_all, ignore_index=True) if closed_all else pd.DataFrame()
openpos = pd.concat(open_all, ignore_index=True) if open_all else pd.DataFrame()

st.subheader("Results")
pnl_view = st.radio(
    "P&L view",
    ["Closed only (realized)", "Closed + Open (separate)"],
    horizontal=True,
)
show_open = pnl_view == "Closed + Open (separate)"

if not closed.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("Realized net P&L (closed only)", f"${closed['net_pnl'].sum():,.2f}")
    c2.metric("Closed matches", f"{len(closed):,}")
    c3.metric("Win rate", f"{(closed['net_pnl'] > 0).mean() * 100:,.1f}%")
else:
    st.warning("No closed trades detected after FIFO matching (you may only have open positions).")

if not openpos.empty:
    side_sign = np.where(openpos["side"] == "SELL", 1.0, -1.0)
    openpos["net_premium"] = (openpos["price"] * openpos["qty"] * openpos["multiplier"] * side_sign) - openpos["fees"]

if show_open:
    if not openpos.empty:
        st.markdown("### Open positions (unrealized, shown separately)")
        o1, o2 = st.columns(2)
        o1.metric("Open premium (cashflow, not P&L)", f"${openpos['net_premium'].sum():,.2f}")
        o2.metric("Open lots", f"{len(openpos):,}")
        st.dataframe(
            openpos.sort_values(["underlying", "expiry", "strike", "right", "side", "date"]),
            use_container_width=True,
        )
    else:
        st.info("No open positions to show separately.")

trades_sheet = closed.copy().reset_index(drop=True) if not closed.empty else pd.DataFrame()
if not trades_sheet.empty:
    trades_sheet["trade_id"] = trades_sheet.index.astype(str)
spy_trades = trades_sheet[trades_sheet["underlying"] == "SPY"].copy() if not trades_sheet.empty else pd.DataFrame()
spy_total = float(spy_trades["net_pnl"].sum()) if not spy_trades.empty else 0.0

# -------------------------
# Insights (out-of-the-box)
# -------------------------
st.subheader("Insights")

if trades_sheet.empty:
    st.info("No closed trades to analyze for insights (only open positions found).")
else:
    # Add derived fields
    ts = trades_sheet.copy()
    ts["open_date"] = ts[["buy_date", "sell_date"]].min(axis=1)
    ts["close_date"] = ts[["buy_date", "sell_date"]].max(axis=1)
    ts["hold_days"] = (ts["close_date"] - ts["open_date"]).dt.total_seconds() / (24 * 3600)
    ts["position_type"] = np.where(ts["sell_date"] < ts["buy_date"], "SHORT", "LONG")

    # Filters
    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            under_sel = st.multiselect("Underlying", options=sorted(ts["underlying"].unique().tolist()),
                                       default=sorted(ts["underlying"].unique().tolist()))
        with c2:
            right_sel = st.multiselect("Type (C/P)", options=sorted(ts["right"].unique().tolist()),
                                       default=sorted(ts["right"].unique().tolist()))
        with c3:
            pos_sel = st.multiselect("Position (SHORT/LONG)", options=sorted(ts["position_type"].unique().tolist()),
                                     default=sorted(ts["position_type"].unique().tolist()))
        with c4:
            dmin = ts["close_date"].min().date()
            dmax = ts["close_date"].max().date()
            date_range = st.date_input("Close date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = date_range
    else:
        d0 = dmin; d1 = dmax

    f = ts[
        ts["underlying"].isin(under_sel)
        & ts["right"].isin(right_sel)
        & ts["position_type"].isin(pos_sel)
        & (ts["close_date"].dt.date >= d0)
        & (ts["close_date"].dt.date <= d1)
    ].copy()

    if f.empty:
        st.warning("No trades match your filters.")
    else:
        wins = f[f["net_pnl"] > 0]["net_pnl"]
        losses = f[f["net_pnl"] < 0]["net_pnl"]

        total_net = float(f["net_pnl"].sum())
        total_gross = float(f["gross_pnl"].sum())
        total_fees = float(f["fees"].sum())
        n = len(f)
        win_rate = float((f["net_pnl"] > 0).mean()) if n else 0.0
        avg = float(f["net_pnl"].mean()) if n else 0.0
        med = float(f["net_pnl"].median()) if n else 0.0
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        payoff = (avg_win / abs(avg_loss)) if (avg_win and avg_loss) else np.nan
        profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) and len(wins) else np.nan
        expectancy = avg  # same as avg net per closed match

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Net P&L (filtered)", f"${total_net:,.2f}")
        c2.metric("Trades (closed matches)", f"{n:,}")
        c3.metric("Win rate", f"{win_rate*100:,.1f}%")
        c4.metric("Expectancy / trade", f"${expectancy:,.2f}")
        c5.metric("Profit factor", "—" if np.isnan(profit_factor) else f"{profit_factor:,.2f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg win", f"${avg_win:,.2f}" if len(wins) else "—")
        c2.metric("Avg loss", f"${avg_loss:,.2f}" if len(losses) else "—")
        c3.metric("Payoff ratio", "—" if np.isnan(payoff) else f"{payoff:,.2f}")
        fee_drag = (total_fees / abs(total_gross)) if total_gross else np.nan
        c4.metric("Fee drag", "—" if np.isnan(fee_drag) else f"{fee_drag*100:,.2f}%")

        # Equity curve + drawdown
        curve = f.sort_values("close_date")[["close_date", "net_pnl"]].copy()
        curve["cum_net"] = curve["net_pnl"].cumsum()
        curve["peak"] = curve["cum_net"].cummax()
        curve["drawdown"] = curve["cum_net"] - curve["peak"]
        max_dd = float(curve["drawdown"].min()) if not curve.empty else 0.0

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Cumulative realized P&L")
            st.line_chart(curve.set_index("close_date")["cum_net"])
            st.caption(f"Max drawdown: ${max_dd:,.2f}")
        with c2:
            st.markdown("#### Holding time")
            st.write(f"Median hold: **{f['hold_days'].median():.1f} days** | Avg hold: **{f['hold_days'].mean():.1f} days**")
            # bucket holds for a quick view
            bins = [-0.01, 1, 3, 7, 14, 30, 90, 3650]
            labels = ["≤1d", "1–3d", "3–7d", "7–14d", "14–30d", "30–90d", "90d+"]
            f["hold_bucket"] = pd.cut(f["hold_days"], bins=bins, labels=labels)
            by_hold = (f.groupby("hold_bucket", as_index=False)
                       .agg(trades=("net_pnl","size"), win_rate=("net_pnl", lambda s: float((s>0).mean())),
                            net_pnl=("net_pnl","sum"), avg_pnl=("net_pnl","mean"))
                       )
            st.dataframe(by_hold, use_container_width=True)

        st.markdown("#### Where you make/lose money")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**By underlying**")
            by_under = (f.groupby("underlying", as_index=False)
                        .agg(trades=("net_pnl","size"),
                             win_rate=("net_pnl", lambda s: float((s>0).mean())),
                             net_pnl=("net_pnl","sum"),
                             avg_pnl=("net_pnl","mean"),
                             fees=("fees","sum"))
                        .sort_values("net_pnl", ascending=False))
            st.dataframe(by_under, use_container_width=True)

        with c2:
            st.markdown("**By option type & position**")
            by_type = (f.groupby(["right","position_type"], as_index=False)
                       .agg(trades=("net_pnl","size"),
                            win_rate=("net_pnl", lambda s: float((s>0).mean())),
                            net_pnl=("net_pnl","sum"),
                            avg_pnl=("net_pnl","mean"))
                       .sort_values(["net_pnl"], ascending=False))
            st.dataframe(by_type, use_container_width=True)

        # Best / worst
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 10 winners**")
            st.dataframe(f.sort_values("net_pnl", ascending=False).head(10)[
                ["close_date","underlying","expiry","strike","right","position_type","qty","net_pnl","fees","hold_days"]
            ], use_container_width=True)
        with c2:
            st.markdown("**Top 10 losers**")
            st.dataframe(f.sort_values("net_pnl", ascending=True).head(10)[
                ["close_date","underlying","expiry","strike","right","position_type","qty","net_pnl","fees","hold_days"]
            ], use_container_width=True)

        # Streaks (based on close order)
        outcomes = (curve["net_pnl"] > 0).tolist()
        def max_streak(vals, target=True):
            best = cur = 0
            for v in vals:
                if v == target:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best
        st.caption(f"Max win streak: **{max_streak(outcomes, True)}** | Max loss streak: **{max_streak(outcomes, False)}**")

# -------------------------
# Mistake detector
# -------------------------
with st.expander("Mistake Detector", expanded=False):
    if trades_sheet.empty:
        st.info("No closed trades to analyze for mistakes.")
    else:
        base_mistakes = trades_sheet.copy()
        base_mistakes["open_date"] = base_mistakes[["buy_date", "sell_date"]].min(axis=1)
        base_mistakes["close_date"] = base_mistakes[["buy_date", "sell_date"]].max(axis=1)

        mistake_output = detect_mistakes(base_mistakes, work)
        mistakes_df = mistake_output["mistakes_df"]
        summary_by_type = mistake_output["summary_by_type"]
        summary_by_week = mistake_output["summary_by_week"]
        examples_by_type = mistake_output["examples_by_type"]
        held_tables = mistake_output["held_tables"]

        if mistakes_df.empty:
            st.info("No mistakes detected with current rules.")
        else:
            top_leak = summary_by_type.iloc[0] if not summary_by_type.empty else None
            over_cnt = int((mistakes_df["mistake_type"] == "Overtrading day").sum())
            big_loss_cnt = int((mistakes_df["mistake_type"] == "Big loss outlier").sum())

            c1, c2, c3 = st.columns(3)
            if top_leak is not None:
                c1.metric("Top leak", f"{top_leak['mistake_type']} (${top_leak['total_damage']:,.2f})")
            else:
                c1.metric("Top leak", "—")
            c2.metric("Overtrading days", f"{over_cnt:,}")
            c3.metric("Big loss outliers", f"{big_loss_cnt:,}")

            st.markdown("### Mistake summary")
            st.dataframe(summary_by_type, use_container_width=True)

            if not summary_by_week.empty:
                st.markdown("### Weekly mistake damage (top 3 types)")
                top_types = summary_by_type.head(3)["mistake_type"].tolist()
                weekly = summary_by_week[summary_by_week["mistake_type"].isin(top_types)]
                weekly_pivot = weekly.pivot(index="week_key", columns="mistake_type", values="total_damage").fillna(0)
                st.line_chart(weekly_pivot)

            st.markdown("### Drilldown")
            type_sel = st.selectbox("Mistake type", summary_by_type["mistake_type"].tolist())
            examples = examples_by_type[examples_by_type["mistake_type"] == type_sel].head(20).copy()
            examples["date"] = examples["close_date"].dt.date
            show_cols = ["date", "underlying", "contract_key", "net_pnl", "severity", "details"]
            st.dataframe(examples[show_cols], use_container_width=True)

            if not examples.empty:
                ex_idx = st.selectbox(
                    "Inspect example row",
                    examples.index.tolist(),
                    format_func=lambda i: f"{examples.loc[i, 'date']} | {examples.loc[i, 'details']}",
                )
                selected = examples.loc[ex_idx]
                trades_view = trades_sheet.copy()
                trades_view["close_date"] = trades_view[["buy_date", "sell_date"]].max(axis=1)
                trades_view["contract_key"] = _contract_key(trades_view)

                if type_sel == "Overtrading day":
                    trades_view = trades_view[trades_view["close_date"].dt.date == selected["date"]]
                elif type_sel == "Gave back gains in ticker":
                    trades_view = trades_view[trades_view["underlying"] == selected["underlying"]]
                elif type_sel == "Big loss outlier":
                    trades_view = trades_view[trades_view["trade_id"] == selected["trade_id"]]
                elif type_sel == "Averaged down (added to loser)":
                    trades_view = trades_view[trades_view["contract_key"] == selected["contract_key"]]
                elif type_sel == "Held losers longer":
                    trades_view = trades_view

                st.markdown("#### Related closed trades")
                st.dataframe(trades_view.sort_values("close_date"), use_container_width=True)

            if "losers" in held_tables and "winners" in held_tables:
                st.markdown("### Holding time contrast")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Longest held losers**")
                    st.dataframe(held_tables["losers"][["close_date", "underlying", "contract_key", "net_pnl", "holding_days"]], use_container_width=True)
                with c2:
                    st.markdown("**Longest held winners**")
                    st.dataframe(held_tables["winners"][["close_date", "underlying", "contract_key", "net_pnl", "holding_days"]], use_container_width=True)

        if trades_sheet[["buy_date", "sell_date"]].isna().any().any():
            st.caption("Need open_date from FIFO pairing to compute holding time.")

# -------------------------
# Actionable recommendations (based on THIS file)
# -------------------------
st.subheader("Post-mortem tags + root-cause stats")

if trades_sheet.empty:
    st.info("No closed trades available to tag yet.")
else:
    if "loss_tags" not in st.session_state:
        st.session_state["loss_tags"] = {}

    losses = trades_sheet[trades_sheet["net_pnl"] < 0].copy()
    if losses.empty:
        st.info("No losing trades found in this file.")
    else:
        losses["position_type"] = np.where(losses["sell_date"] < losses["buy_date"], "SHORT", "LONG")
        right_label = losses["right"].map({"C": "Call", "P": "Put"}).fillna(losses["right"])
        losses["strategy"] = right_label + " " + losses["position_type"].str.title()
        losses["post_mortem_tag"] = losses["trade_id"].map(st.session_state["loss_tags"]).fillna(UNASSIGNED_TAG)

        edit_cols = [
            "close_date",
            "underlying",
            "expiry",
            "strike",
            "right",
            "position_type",
            "qty",
            "net_pnl",
            "post_mortem_tag",
        ]
        edited = st.data_editor(
            losses.set_index("trade_id")[edit_cols],
            use_container_width=True,
            hide_index=True,
            key="loss_tag_editor",
            disabled=[c for c in edit_cols if c != "post_mortem_tag"],
            column_config={
                "post_mortem_tag": st.column_config.SelectboxColumn(
                    "Post-mortem tag",
                    options=[UNASSIGNED_TAG] + POST_MORTEM_TAGS,
                )
            },
        )

        for trade_id, tag in edited["post_mortem_tag"].to_dict().items():
            if tag and tag != UNASSIGNED_TAG:
                st.session_state["loss_tags"][trade_id] = tag
            else:
                st.session_state["loss_tags"].pop(trade_id, None)

        tagged_losses = losses.drop(columns=["post_mortem_tag"]).merge(
            edited[["post_mortem_tag"]],
            left_on="trade_id",
            right_index=True,
            how="left",
        )
        tagged_losses["post_mortem_tag"] = tagged_losses["post_mortem_tag"].fillna(UNASSIGNED_TAG)
        tagged_losses["loss_usd"] = -tagged_losses["net_pnl"]
        tagged = tagged_losses[tagged_losses["post_mortem_tag"] != UNASSIGNED_TAG].copy()

        st.markdown("### Root-cause summary (losing trades)")
        unassigned = int((tagged_losses["post_mortem_tag"] == UNASSIGNED_TAG).sum())
        if unassigned:
            st.caption(f"{unassigned} losing trades are still untagged.")

        if tagged.empty:
            st.info("Assign tags above to see root-cause stats.")
        else:
            summary = (
                tagged.groupby("post_mortem_tag", as_index=False)
                .agg(
                    loss_usd=("loss_usd", "sum"),
                    count=("net_pnl", "size"),
                    avg_loss=("net_pnl", "mean"),
                    tickers=("underlying", lambda s: ", ".join(sorted(set(s)))),
                    strategies=("strategy", lambda s: ", ".join(sorted(set(s)))),
                )
                .sort_values("loss_usd", ascending=False)
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top 3 loss causes by $**")
                st.dataframe(summary.head(3), use_container_width=True)
            with c2:
                st.markdown("**Top 3 loss causes by count**")
                by_count = summary.sort_values(["count", "loss_usd"], ascending=[False, False])
                st.dataframe(by_count.head(3), use_container_width=True)

            st.markdown("**Average loss per cause + tickers/strategies**")
            st.dataframe(summary, use_container_width=True)

# -------------------------
# Actionable recommendations (based on THIS file)
# -------------------------
st.subheader("Actionable recommendations")

if trades_sheet.empty:
    st.info("No closed trades to generate recommendations.")
else:
    base = trades_sheet.copy()
    base["open_date"] = base[["buy_date", "sell_date"]].min(axis=1)
    base["close_date"] = base[["buy_date", "sell_date"]].max(axis=1)
    base["hold_days"] = (base["close_date"] - base["open_date"]).dt.total_seconds() / (24 * 3600)
    base["position_type"] = np.where(base["sell_date"] < base["buy_date"], "SHORT", "LONG")

    use_filters_for_recos = st.toggle("Use the Insights filters above for these recommendations", value=False)
    data = f.copy() if (use_filters_for_recos and "f" in locals() and isinstance(locals().get("f"), pd.DataFrame) and not locals().get("f").empty) else base

    # 1) Intraday churn & fee drag
    intraday = data[data["hold_days"] <= 1.0].copy()
    allcnt = len(data)
    icnt = len(intraday)

    intraday_net = float(intraday["net_pnl"].sum()) if icnt else 0.0
    intraday_gross = float(intraday["gross_pnl"].sum()) if icnt else 0.0
    intraday_fees = float(intraday["fees"].sum()) if icnt else 0.0

    # 2) Segment performance: underlying x (right, position_type)
    seg = (data.groupby(["underlying", "right", "position_type"], as_index=False)
           .agg(trades=("net_pnl", "size"),
                win_rate=("net_pnl", lambda s: float((s > 0).mean())),
                net_pnl=("net_pnl", "sum"),
                avg_pnl=("net_pnl", "mean"),
                fees=("fees", "sum"))
           .sort_values("net_pnl", ascending=True))
    seg = seg.copy()
    right_label = seg["right"].map({"C": "Call", "P": "Put"}).fillna(seg["right"])
    seg["segment"] = seg["underlying"] + " " + right_label + " " + seg["position_type"].str.title()
    seg["segment_key"] = seg["underlying"] + "|" + seg["right"] + "|" + seg["position_type"]

    # 3) Size buckets
    def size_bucket(q):
        q = int(q)
        if q == 1:
            return "1"
        if q == 2:
            return "2"
        if 3 <= q <= 5:
            return "3–5"
        if 6 <= q <= 10:
            return "6–10"
        return "11+"

    sb = data.copy()
    sb["size_bucket"] = sb["qty"].apply(size_bucket)
    by_size = (sb.groupby("size_bucket", as_index=False)
               .agg(trades=("net_pnl","size"),
                    win_rate=("net_pnl", lambda s: float((s>0).mean())),
                    net_pnl=("net_pnl","sum"),
                    avg_pnl=("net_pnl","mean"),
                    fees=("fees","sum"))
               )
    order = ["1","2","3–5","6–10","11+"]
    by_size["size_bucket"] = pd.Categorical(by_size["size_bucket"], categories=order, ordered=True)
    by_size = by_size.sort_values("size_bucket")

    # -------- Present key findings --------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Closed matches analyzed", f"{allcnt:,}")
    c2.metric("Intraday share", f"{(icnt/allcnt*100 if allcnt else 0):,.1f}%")
    c3.metric("Intraday net P&L", f"${intraday_net:,.2f}")
    c4.metric("Intraday fees", f"${intraday_fees:,.2f}")

    st.markdown("### Key diagnostics")

    # Heuristic flags
    reco_lines = []

    if icnt >= max(10, 0.4 * allcnt):
        # fee-driven churn check
        if abs(intraday_gross) <= 0.2 * intraday_fees and intraday_fees > 0:
            reco_lines.append(f"**Reduce intraday churn.** Intraday gross is near flat (${intraday_gross:,.2f}) but fees are ${intraday_fees:,.2f}, turning it into net ${intraday_net:,.2f}.")
        elif intraday_net < 0:
            reco_lines.append(f"**Intraday segment is negative.** Net ${intraday_net:,.2f} across {icnt} intraday matches — consider banning 0–1 day holds for a week and re-checking expectancy.")
    elif icnt > 0 and intraday_net < 0:
        reco_lines.append(f"**Intraday is negative** (${intraday_net:,.2f}). Consider moving to 2–7 day holds or fewer round trips.")

    # Identify worst segment(s) with enough samples
    seg2 = seg[seg["trades"] >= 10].copy()
    if not seg2.empty:
        worst = seg2.iloc[0]
        reco_lines.append(f"**Stop/Reduce this losing segment:** {worst['underlying']} {worst['position_type']} {('Calls' if worst['right']=='C' else 'Puts')} — net ${float(worst['net_pnl']):,.2f} over {int(worst['trades'])} matches (win rate {float(worst['win_rate'])*100:,.1f}%).")

    # SPY focus if present
    spy = seg[seg["underlying"] == "SPY"].copy()
    if not spy.empty:
        spy_lc = spy[(spy["right"]=="C") & (spy["position_type"]=="LONG")]
        if not spy_lc.empty and float(spy_lc["net_pnl"].iloc[0]) < 0 and int(spy_lc["trades"].iloc[0]) >= 5:
            reco_lines.append(f"**SPY long calls are hurting you:** net ${float(spy_lc['net_pnl'].iloc[0]):,.2f} over {int(spy_lc['trades'].iloc[0])} matches. Consider avoiding same-day SPY call buys or switching to longer DTE / defined-risk structures.")

        spy_sp = spy[(spy["right"]=="P") & (spy["position_type"]=="SHORT")]
        if not spy_sp.empty and float(spy_sp["net_pnl"].iloc[0]) > 0 and int(spy_sp["trades"].iloc[0]) >= 3:
            reco_lines.append(f"**Lean into what works:** SPY short puts show net ${float(spy_sp['net_pnl'].iloc[0]):,.2f} over {int(spy_sp['trades'].iloc[0])} matches. Keep this style; add an exit rule (e.g., take profit at 50–70% of premium) to reduce tail losses.")

    # Oversizing
    lose_big = by_size[(by_size["size_bucket"].isin(["6–10","11+"])) & (by_size["avg_pnl"] < 0)]
    if not lose_big.empty:
        worstb = lose_big.sort_values("avg_pnl").iloc[0]
        reco_lines.append(f"**Cap size on this bucket:** {worstb['size_bucket']} contracts have avg ${float(worstb['avg_pnl']):,.2f}/match (net ${float(worstb['net_pnl']):,.2f}). Cap longs to ≤5 until this flips positive.")

    if not reco_lines:
        reco_lines.append("No strong red flags detected with current heuristics. Use the tables below to decide what to scale up/down.")

    for line in reco_lines:
        st.warning(line)

    st.markdown("### Tables you can act on")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Worst segments (min 5 trades)**")
        st.dataframe(seg[seg["trades"] >= 5].sort_values("net_pnl").head(20), use_container_width=True)
    with c2:
        st.markdown("**By size bucket**")
        st.dataframe(by_size, use_container_width=True)

    st.markdown("### Quick “Do more / Do less” lists")
    base_seg = seg[seg["trades"] >= 3].copy()
    top = base_seg.sort_values("net_pnl", ascending=False).head(10)
    bot = base_seg.sort_values("net_pnl", ascending=True).head(10)
    overlap_keys = set(top["segment_key"]).intersection(bot["segment_key"])

    def add_list_flags(df: pd.DataFrame, list_name: str) -> pd.DataFrame:
        flagged = df.copy()
        flagged["list_flag"] = np.where(flagged["segment_key"].isin(overlap_keys), "both lists", list_name)
        return flagged

    top = add_list_flags(top, "top")
    bot = add_list_flags(bot, "bottom")
    c1, c2 = st.columns(2)
    with c1:
        st.success("Do more of (best segments)")
        top_show = top.drop(columns=["segment_key"])
        top_show = top_show[["segment", "list_flag", "trades", "win_rate", "net_pnl", "avg_pnl", "fees",
                             "underlying", "right", "position_type"]]
        st.dataframe(top_show, use_container_width=True)
    with c2:
        st.error("Do less of (worst segments)")
        bot_show = bot.drop(columns=["segment_key"])
        bot_show = bot_show[["segment", "list_flag", "trades", "win_rate", "net_pnl", "avg_pnl", "fees",
                             "underlying", "right", "position_type"]]
        st.dataframe(bot_show, use_container_width=True)

    if overlap_keys:
        st.caption("Segments tagged as “both lists” appear because the list size exceeds the number of unique segments "
                   "or because net P&L is near the middle of the distribution. Use trades, win rate, and avg P&L to "
                   "decide whether the segment is truly strong or weak.")





if not trades_sheet.empty:
    dash = (trades_sheet.groupby("underlying", as_index=False)
            .agg(net_pnl=("net_pnl", "sum"), gross_pnl=("gross_pnl", "sum"), fees=("fees", "sum"))
            ).sort_values("net_pnl", ascending=False)
    st.markdown("### Dashboard (realized)")
    st.dataframe(dash, use_container_width=True)

st.subheader("Export Excel")
fname = st.text_input("Output filename", "trade_analysis.xlsx")

def to_excel_bytes():
    import io as _io
    from openpyxl.styles import PatternFill, Font
    from openpyxl.utils import get_column_letter

    buf = _io.BytesIO()

    dash = pd.DataFrame()
    if not trades_sheet.empty:
        dash = (trades_sheet.groupby("underlying", as_index=False)
                .agg(net_pnl=("net_pnl", "sum"), gross_pnl=("gross_pnl", "sum"), fees=("fees", "sum"))
                ).sort_values("net_pnl", ascending=False)

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if not dash.empty:
            dash.to_excel(writer, sheet_name="Dashboard", index=False)
        if not trades_sheet.empty:
            trades_sheet.to_excel(writer, sheet_name="Trades", index=False)
        if not spy_trades.empty:
            spy_trades.to_excel(writer, sheet_name="SPY", index=False)
        if not openpos.empty:
            openpos.to_excel(writer, sheet_name="OpenPositions", index=False)
        
        # Insights summary
        if not trades_sheet.empty:
            _base = trades_sheet.copy()
            _base["open_date"] = _base[["buy_date", "sell_date"]].min(axis=1)
            _base["close_date"] = _base[["buy_date", "sell_date"]].max(axis=1)
            _base["hold_days"] = (_base["close_date"] - _base["open_date"]).dt.total_seconds() / (24 * 3600)
            _base["position_type"] = np.where(_base["sell_date"] < _base["buy_date"], "SHORT", "LONG")
            _intraday = _base[_base["hold_days"] <= 1.0]
            intraday_summary = pd.DataFrame([{
                "closed_matches": len(_base),
                "intraday_matches": len(_intraday),
                "intraday_share": (len(_intraday)/len(_base) if len(_base) else 0.0),
                "intraday_net_pnl": float(_intraday["net_pnl"].sum()) if len(_intraday) else 0.0,
                "intraday_gross_pnl": float(_intraday["gross_pnl"].sum()) if len(_intraday) else 0.0,
                "intraday_fees": float(_intraday["fees"].sum()) if len(_intraday) else 0.0,
            }])
            intraday_summary.to_excel(writer, sheet_name="Insights", index=False)

            seg = (_base.groupby(["underlying","right","position_type"], as_index=False)
                   .agg(trades=("net_pnl","size"),
                        win_rate=("net_pnl", lambda s: float((s>0).mean())),
                        net_pnl=("net_pnl","sum"),
                        avg_pnl=("net_pnl","mean"),
                        fees=("fees","sum"))
                   .sort_values("net_pnl", ascending=True))
            seg.to_excel(writer, sheet_name="Segments", index=False)

        work.to_excel(writer, sheet_name="RawParsed", index=False)

        wb = writer.book

        if "Trades" in wb.sheetnames:
            ws = wb["Trades"]
            headers = [cell.value for cell in ws[1]]
            if "underlying" in headers:
                idx = headers.index("underlying") + 1
                fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
                for r in range(2, ws.max_row + 1):
                    if ws.cell(row=r, column=idx).value == "SPY":
                        for c in range(1, ws.max_column + 1):
                            ws.cell(row=r, column=c).fill = fill
            for col in range(1, min(ws.max_column, 20) + 1):
                ws.column_dimensions[get_column_letter(col)].width = 16

        if "SPY" in wb.sheetnames:
            ws = wb["SPY"]
            last = ws.max_row + 2
            ws.cell(row=last, column=1).value = "SPY net total (closed)"
            ws.cell(row=last, column=2).value = spy_total
            ws.cell(row=last, column=1).font = Font(bold=True)
            ws.cell(row=last, column=2).font = Font(bold=True)

    buf.seek(0)
    return buf.getvalue()

if st.button("Generate Excel"):
    st.download_button(
        "Download Excel",
        data=to_excel_bytes(),
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.success("Generated Excel.")
