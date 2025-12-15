# app.py
import io
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ============================
# STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="Decisional Dashboard", layout="wide")
st.title("Decisional Dashboard (PKRV + Monthly Revenue Model)")

# ============================
# CONSTANTS
# ============================
# Google Drive File ID (from your link)
FILE_ID = "1GpZO2zaLaeKrZwt-ttNx59ipskx_oiuG"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Sheet names aligned to your provided workbook structure
SHEET_MONTHLY_REVENUE = "Monthly Revenue"
SHEET_EXPENDITURES = "Expenditures"

# Optional / if present
SHEET_KPIS = "KPIs"          # if you later add a simple Metric/Value table
SHEET_BENCH = "Benchmark"    # optional
SHEET_WAM = "WAM"            # optional

# ============================
# GOOGLE DRIVE: DOWNLOAD/EXPORT
# ============================
@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_drive_service():
    """
    Expects Streamlit secrets:
      - st.secrets["gdrive_service_account"] as service account JSON dict
    """
    creds_info = st.secrets["gdrive_service_account"]
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)

@st.cache_data(ttl=60 * 30, show_spinner=True)
def download_drive_file_as_xlsx(file_id: str) -> bytes:
    """
    Robust for native Google Sheets:
      - Exports as XLSX
      - Returns bytes readable by openpyxl
    """
    service = get_drive_service()
    request = service.files().export_media(
        fileId=file_id,
        mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()

@st.cache_data(ttl=60 * 30, show_spinner=True)
def load_workbook_sheets(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    bio = io.BytesIO(file_bytes)

    # Try to read the required sheets plus optional ones if they exist
    # (pandas raises if a requested sheet doesn't exist, so we discover first)
    xls = pd.ExcelFile(bio, engine="openpyxl")
    available = set(xls.sheet_names)

    wanted = [SHEET_MONTHLY_REVENUE, SHEET_EXPENDITURES, SHEET_KPIS, SHEET_BENCH, SHEET_WAM]
    to_read = [s for s in wanted if s in available]

    dfs = pd.read_excel(bio, sheet_name=to_read, engine="openpyxl", header=None)

    # normalize keys + ensure dict return
    if isinstance(dfs, pd.DataFrame):
        dfs = {to_read[0]: dfs}

    return dfs

# ============================
# MUFAP PKRV INGEST (PLACEHOLDER)
# ============================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_mufap_pkrv_history(path_or_url: str) -> pd.DataFrame:
    """
    Placeholder loader. Replace with MUFAP fetch logic when ready.
    Expected columns: Date, Tenor, Yield
    """
    df = pd.read_csv(path_or_url)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Tenor"] = df["Tenor"].astype(str).str.strip()
    df["Yield"] = pd.to_numeric(df["Yield"], errors="coerce")
    df = df.dropna(subset=["Date", "Tenor", "Yield"])
    return df

def pkrv_curve_for_date(df: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    out = df.loc[df["Date"] == d, ["Tenor", "Yield"]].copy()
    tenor_order = ["1M","3M","6M","9M","1Y","2Y","3Y","4Y","5Y","7Y","10Y","15Y","20Y","30Y"]
    out["Tenor"] = pd.Categorical(out["Tenor"], categories=tenor_order, ordered=True)
    return out.sort_values("Tenor")

# ============================
# PARSERS: MONTHLY REVENUE / EXPENDITURES (as per provided workbook style)
# ============================
def _find_date_header_row(df_raw: pd.DataFrame) -> tuple[int, list[int], list[pd.Timestamp]]:
    """
    Finds first row containing datetime-like objects. Returns:
      (row_index, date_col_indices, dates_as_timestamp)
    """
    date_row = None
    for r in range(df_raw.shape[0]):
        for c in range(df_raw.shape[1]):
            v = df_raw.iat[r, c]
            if isinstance(v, (dt.datetime, pd.Timestamp)):
                date_row = r
                break
        if date_row is not None:
            break

    if date_row is None:
        return -1, [], []

    date_cols = []
    dates = []
    for c in range(df_raw.shape[1]):
        v = df_raw.iat[date_row, c]
        if isinstance(v, (dt.datetime, pd.Timestamp)):
            date_cols.append(c)
            dates.append(pd.to_datetime(v))

    return date_row, date_cols, dates

def parse_monthly_revenue(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly Revenue sheet (structured blocks with dates across columns).
    Produces a tidy monthly revenue series with a conservative definition:
      Revenue = (sum of all 'Net Profit' rows) + (MTBs 'Total' row)
    This matches the sheet’s primary income blocks (placements + MTBs).
    """
    date_row, date_cols, dates = _find_date_header_row(df_raw)
    if date_row < 0 or not date_cols:
        return pd.DataFrame(columns=["Date", "Revenue", "Placements_NetProfit", "MTBs_Total"])

    # labels assumed in column 1 in the workbook
    labels = df_raw.iloc[:, 1].astype(str).str.strip().str.lower()

    net_profit_rows = np.where(labels == "net profit")[0].tolist()

    # MTBs total row: first row whose label == "total" AFTER the row labeled "mtbs"
    mtbs_row_candidates = np.where(labels == "mtbs")[0].tolist()
    mtbs_total_row = None
    if mtbs_row_candidates:
        mtbs_start = mtbs_row_candidates[0]
        after = labels.iloc[mtbs_start + 1 :]
        total_after = after[after == "total"]
        if not total_after.empty:
            mtbs_total_row = int(total_after.index[0])

    # extract values
    def _row_sum(rows: list[int]) -> np.ndarray:
        if not rows:
            return np.zeros(len(date_cols), dtype=float)
        arr = np.zeros(len(date_cols), dtype=float)
        for r in rows:
            vals = pd.to_numeric(df_raw.iloc[r, date_cols], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            arr += vals
        return arr

    placements_np = _row_sum(net_profit_rows)
    mtbs_total = np.zeros(len(date_cols), dtype=float)
    if mtbs_total_row is not None:
        mtbs_total = pd.to_numeric(df_raw.iloc[mtbs_total_row, date_cols], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "Date": dates,
            "Placements_NetProfit": placements_np,
            "MTBs_Total": mtbs_total,
        }
    )
    out["Revenue"] = out["Placements_NetProfit"] + out["MTBs_Total"]
    out = out.sort_values("Date").reset_index(drop=True)
    return out

def parse_expenditures(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expenditures sheet parser aligned to the provided workbook style:
      - categories in column 1
      - dates across columns (same approach)
    Returns tidy: Date, CostCenter, Amount
    """
    date_row, date_cols, dates = _find_date_header_row(df_raw)
    if date_row < 0 or not date_cols:
        return pd.DataFrame(columns=["Date", "CostCenter", "Amount"])

    name_col = 1  # as per workbook layout
    rows = []

    for r in range(date_row + 1, df_raw.shape[0]):
        name = df_raw.iat[r, name_col]
        if pd.isna(name):
            continue
        name = str(name).strip()
        if not name:
            continue

        vals = pd.to_numeric(df_raw.iloc[r, date_cols], errors="coerce")
        if vals.notna().sum() == 0:
            continue

        for c_idx, d in zip(date_cols, dates):
            amt = pd.to_numeric(df_raw.iat[r, c_idx], errors="coerce")
            if pd.isna(amt):
                continue
            rows.append((pd.to_datetime(d), name, float(amt)))

    out = pd.DataFrame(rows, columns=["Date", "CostCenter", "Amount"])
    out = out.sort_values(["Date", "CostCenter"]).reset_index(drop=True)
    return out

# ============================
# HELPERS: KPIs (optional if you add Metric/Value later)
# ============================
def get_scalar_kpi_from_table(df_any: pd.DataFrame, metric_name: str, metric_col_idx: int = 0, value_col_idx: int = 1):
    """
    If you later add a simple KPI sheet in two columns (Metric | Value),
    this reads it safely even when loaded with header=None.
    """
    if df_any is None or df_any.empty:
        return None
    m = df_any.iloc[:, metric_col_idx].astype(str).str.strip().str.lower()
    hit = df_any.loc[m == metric_name.lower()]
    if hit.empty:
        return None
    val = pd.to_numeric(hit.iloc[0, value_col_idx], errors="coerce")
    return None if pd.isna(val) else float(val)

# ============================
# SIDEBAR CONTROLS
# ============================
with st.sidebar:
    st.subheader("Inputs")
    mufap_source = st.text_input("MUFAP PKRV history CSV path/URL", value="data/pkrv_history.csv")
    st.caption("PKRV loader is a placeholder (CSV). Replace with MUFAP download + cache when ready.")

    st.divider()
    st.subheader("Controls")
    lookback_days = st.slider("PKRV lookback (days)", 30, 365, 180)

# ============================
# LOAD WORKBOOK (Drive) + PARSE
# ============================
try:
    xlsx_bytes = download_drive_file_as_xlsx(FILE_ID)
    sheets = load_workbook_sheets(xlsx_bytes)
except Exception as e:
    st.error(
        "Unable to load the Google Drive workbook.\n\n"
        "Common causes:\n"
        "  1) The Google Sheet is not shared with the service account email.\n"
        "  2) st.secrets['gdrive_service_account'] is missing/invalid.\n"
        "  3) Sheet names do not match.\n\n"
        f"Error: {e}"
    )
    st.stop()

df_monthly_revenue_raw = sheets.get(SHEET_MONTHLY_REVENUE, pd.DataFrame())
df_expenditures_raw = sheets.get(SHEET_EXPENDITURES, pd.DataFrame())
df_kpis_raw = sheets.get(SHEET_KPIS, pd.DataFrame())  # optional

rev = parse_monthly_revenue(df_monthly_revenue_raw)
exp = parse_expenditures(df_expenditures_raw)

# ============================
# LOAD PKRV (CSV placeholder)
# ============================
try:
    df_pkrv = load_mufap_pkrv_history(mufap_source)
except Exception as e:
    st.warning(f"Could not load MUFAP PKRV history from '{mufap_source}'. Error: {e}")
    df_pkrv = pd.DataFrame(columns=["Date", "Tenor", "Yield"])

# ============================
# KPI CALCS (Latest Month)
# ============================
latest_month = None
revenue_latest = None
cost_latest = None
cti_latest = None

if not rev.empty:
    latest_month = rev["Date"].max()
    revenue_latest = float(rev.loc[rev["Date"] == latest_month, "Revenue"].iloc[0])

if not exp.empty and latest_month is not None:
    # costs for the same month-end date (sheet uses month-end dates)
    cost_latest = float(exp.loc[exp["Date"] == latest_month, "Amount"].sum())
    if revenue_latest and revenue_latest != 0:
        cti_latest = 100.0 * cost_latest / revenue_latest

# Benchmark yield + WAM:
# If you later add a KPI table, it will be picked up here; otherwise card shows "—"
benchmark_yield = get_scalar_kpi_from_table(df_kpis_raw, "BenchmarkYield")
wam_years = get_scalar_kpi_from_table(df_kpis_raw, "WeightedAverageMaturityYears")

# ============================
# PKRV SLICES
# ============================
p10y = None
today_curve = pd.DataFrame(columns=["Tenor", "Yield"])
df_pkrv_lb = pd.DataFrame(columns=["Date", "Tenor", "Yield"])
max_date = None

if not df_pkrv.empty:
    max_date = df_pkrv["Date"].max()
    min_date = max_date - pd.Timedelta(days=lookback_days)
    df_pkrv_lb = df_pkrv[df_pkrv["Date"].between(min_date, max_date)].copy()
    today_curve = pkrv_curve_for_date(df_pkrv, max_date)

    hit10 = today_curve.loc[today_curve["Tenor"].astype(str).str.upper() == "10Y", "Yield"]
    if not hit10.empty:
        p10y = float(hit10.iloc[0])

# ============================
# KPI CARDS
# ============================
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Benchmark Yield", f"{benchmark_yield:.2f}%" if benchmark_yield is not None else "—")
c2.metric("Weighted Avg. Maturity (Years)", f"{wam_years:.2f}" if wam_years is not None else "—")
c3.metric("Cost-to-Income Ratio", f"{cti_latest:.1f}%" if cti_latest is not None else "—")
c4.metric("Revenue (Latest Month)", f"{revenue_latest:,.0f}" if revenue_latest is not None else "—")
c5.metric("PKRV 10Y (Latest)", f"{p10y:.2f}%" if p10y is not None else "—")

st.divider()

# ============================
# REVENUE CHART (Monthly Revenue sheet)
# ============================
st.subheader("Monthly Revenue (from 'Monthly Revenue' sheet)")

if rev.empty:
    st.warning(f"Sheet '{SHEET_MONTHLY_REVENUE}' was not parsed into a revenue series. Check that the date row exists.")
else:
    rev_chart = alt.Chart(rev).mark_line(point=True).encode(
        x=alt.X("Date:T", title="Month"),
        y=alt.Y("Revenue:Q", title="Revenue"),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Revenue:Q", format=",.0f")]
    ).properties(height=320)
    st.altair_chart(rev_chart, use_container_width=True)

    with st.expander("Revenue components (debug / transparency)"):
        st.dataframe(rev, use_container_width=True, height=260)

st.divider()

# ============================
# COST CENTERS (Expenditures sheet)
# ============================
st.subheader("Cost Centers (from 'Expenditures' sheet)")

if exp.empty:
    st.warning(f"Sheet '{SHEET_EXPENDITURES}' was not parsed into cost-center time series. Check that the date row exists.")
else:
    # Filter to latest month by default
    months = sorted(exp["Date"].dropna().unique().tolist())
    default_month = months[-1] if months else None
    pick_month = st.selectbox("Select Month", months, index=len(months) - 1 if months else 0, format_func=lambda d: pd.to_datetime(d).date())

    exp_m = exp[exp["Date"] == pick_month].copy()
    exp_m = exp_m.sort_values("Amount", ascending=False)

    st.dataframe(exp_m, use_container_width=True, height=320)

    # Monthly stacked chart (top N cost centers)
    top_n = st.slider("Top cost centers to chart", 5, 30, 12)
    top_centers = (
        exp.groupby("CostCenter")["Amount"].sum().sort_values(ascending=False).head(top_n).index.tolist()
    )
    exp_top = exp[exp["CostCenter"].isin(top_centers)].copy()

    cost_chart = alt.Chart(exp_top).mark_bar().encode(
        x=alt.X("Date:T", title="Month"),
        y=alt.Y("sum(Amount):Q", title="Cost"),
        color=alt.Color("CostCenter:N"),
        tooltip=[alt.Tooltip("Date:T"), "CostCenter:N", alt.Tooltip("sum(Amount):Q", format=",.0f")]
    ).properties(height=360)
    st.altair_chart(cost_chart, use_container_width=True)

st.divider()

# ============================
# PKRV VISUALS
# ============================
st.subheader("PKRV (MUFAP source - placeholder loader)")

left, right = st.columns([1.2, 1])

with left:
    st.markdown("**PKRV Curve (Latest)**")
    if max_date is None or today_curve.empty:
        st.info("No PKRV curve available. Load a valid PKRV history CSV with columns: Date, Tenor, Yield.")
    else:
        curve_chart = alt.Chart(today_curve).mark_line(point=True).encode(
            x=alt.X("Tenor:N", sort=None),
            y=alt.Y("Yield:Q"),
            tooltip=["Tenor", alt.Tooltip("Yield:Q", format=".2f")]
        ).properties(height=300)
        st.altair_chart(curve_chart, use_container_width=True)
        st.caption(f"Latest PKRV date: {pd.to_datetime(max_date).date()}")

with right:
    st.markdown("**PKRV History (Selected Tenors)**")
    if df_pkrv_lb.empty:
        st.info("PKRV history not available (or empty after lookback filter).")
    else:
        tenors = sorted(df_pkrv_lb["Tenor"].unique().tolist())
        default_tenors = [t for t in ["1Y", "3Y", "5Y", "10Y"] if t in tenors]
        pick = st.multiselect("Tenors", tenors, default=default_tenors)

        hist = df_pkrv_lb[df_pkrv_lb["Tenor"].isin(pick)].copy()
        if hist.empty:
            st.info("Select at least one tenor with available history.")
        else:
            hist_chart = alt.Chart(hist).mark_line().encode(
                x="Date:T",
                y="Yield:Q",
                color="Tenor:N",
                tooltip=["Date:T", "Tenor:N", alt.Tooltip("Yield:Q", format=".2f")]
            ).properties(height=300)
            st.altair_chart(hist_chart, use_container_width=True)

st.caption(
    "Operational note: Ensure the Google Sheet is shared with the service account email "
    "(st.secrets['gdrive_service_account']['client_email'])."
)
