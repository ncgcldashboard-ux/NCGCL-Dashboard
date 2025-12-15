# app.py
import io
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
st.title("Decisional Dashboard (PKRV + Monthly Finance Pack)")

# ============================
# CONSTANTS
# ============================
# Google Drive File ID (extracted from your link)
FILE_ID = "1GpZO2zaLaeKrZwt-ttNx59ipskx_oiuG"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Workbook sheet names (adjust to match your file tabs exactly)
SHEET_KPIS = "KPIs"                  # expected columns: Metric, Value
SHEET_COST_CENTERS = "CostCenters"   # expected columns: Month, CostCenter, Amount
SHEET_BENCH = "Benchmark"            # optional
SHEET_WAM = "WAM"                    # optional

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
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)

@st.cache_data(ttl=60 * 30, show_spinner=True)
def download_drive_file_as_xlsx(file_id: str) -> bytes:
    """
    Robust approach for Google Sheets:
      - Exports the file as XLSX (even if it’s a native Google Sheet)
      - Returns bytes suitable for pandas/openpyxl
    """
    service = get_drive_service()
    request = service.files().export_media(
        fileId=file_id,
        mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    return fh.getvalue()

@st.cache_data(ttl=60 * 30, show_spinner=True)
def load_monthly_workbook(file_bytes: bytes) -> dict[str, pd.DataFrame]:
    bio = io.BytesIO(file_bytes)
    dfs = pd.read_excel(
        bio,
        sheet_name=[SHEET_KPIS, SHEET_COST_CENTERS, SHEET_BENCH, SHEET_WAM],
        engine="openpyxl"
    )
    # normalize column names
    for k, df in dfs.items():
        df.columns = [str(c).strip() for c in df.columns]
    return dfs

# ============================
# MUFAP PKRV INGEST (PLACEHOLDER)
# ============================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_mufap_pkrv_history(path_or_url: str) -> pd.DataFrame:
    """
    Placeholder loader. Replace with MUFAP fetch logic when ready.
    Expected columns:
      - Date (YYYY-MM-DD)
      - Tenor (e.g., 3M, 6M, 1Y, 3Y, 5Y, 10Y, 15Y, 20Y, 30Y)
      - Yield (numeric)
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
# HELPERS: KPIs
# ============================
def get_scalar_kpi(df: pd.DataFrame, metric_name: str, metric_col: str = "Metric", value_col: str = "Value"):
    if df is None or df.empty:
        return None
    if metric_col not in df.columns or value_col not in df.columns:
        return None
    hit = df.loc[
        df[metric_col].astype(str).str.strip().str.lower() == metric_name.lower(),
        value_col
    ]
    if hit.empty:
        return None
    val = pd.to_numeric(hit.iloc[0], errors="coerce")
    return None if pd.isna(val) else float(val)

# ============================
# SIDEBAR CONTROLS
# ============================
with st.sidebar:
    st.subheader("Inputs")
    mufap_source = st.text_input("MUFAP PKRV history CSV path/URL", value="data/pkrv_history.csv")
    st.caption("This is a placeholder. Replace with a MUFAP downloader + local cache when ready.")

    st.divider()
    st.subheader("Controls")
    lookback_days = st.slider("PKRV lookback (days)", 30, 365, 180)

# ============================
# LOAD DATA
# ============================
try:
    xlsx_bytes = download_drive_file_as_xlsx(FILE_ID)
    wb = load_monthly_workbook(xlsx_bytes)
except Exception as e:
    st.error(
        "Unable to load the Google Drive workbook. "
        "Common causes: (1) the file is not shared with the service account email, "
        "(2) secrets are missing/invalid, or (3) sheet names do not match.\n\n"
        f"Error: {e}"
    )
    st.stop()

df_kpis = wb.get(SHEET_KPIS, pd.DataFrame())
df_cost = wb.get(SHEET_COST_CENTERS, pd.DataFrame())
df_bench = wb.get(SHEET_BENCH, pd.DataFrame())
df_wam = wb.get(SHEET_WAM, pd.DataFrame())

try:
    df_pkrv = load_mufap_pkrv_history(mufap_source)
except Exception as e:
    st.warning(f"Could not load MUFAP PKRV history from '{mufap_source}'. Error: {e}")
    df_pkrv = pd.DataFrame(columns=["Date", "Tenor", "Yield"])

# ============================
# EXTRACT KPI VALUES (from Drive workbook)
# ============================
benchmark_yield = get_scalar_kpi(df_kpis, "BenchmarkYield")
cti_ratio = get_scalar_kpi(df_kpis, "CostToIncomeRatio")
wam_years = get_scalar_kpi(df_kpis, "WeightedAverageMaturityYears")

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
c1, c2, c3, c4 = st.columns(4)
c1.metric("Benchmark Yield", f"{benchmark_yield:.2f}%" if benchmark_yield is not None else "—")
c2.metric("Weighted Avg. Maturity (Years)", f"{wam_years:.2f}" if wam_years is not None else "—")
c3.metric("Cost-to-Income Ratio", f"{cti_ratio:.1f}%" if cti_ratio is not None else "—")
c4.metric("PKRV 10Y (Latest)", f"{p10y:.2f}%" if p10y is not None else "—")

st.divider()

# ============================
# CHARTS: PKRV CURVE + HISTORY
# ============================
left, right = st.columns([1.2, 1])

with left:
    st.subheader("PKRV Curve (Latest)")
    if max_date is None or today_curve.empty:
        st.info("No PKRV curve available. Load a valid PKRV history CSV (Date, Tenor, Yield).")
    else:
        chart = alt.Chart(today_curve).mark_line(point=True).encode(
            x=alt.X("Tenor:N", sort=None),
            y=alt.Y("Yield:Q"),
            tooltip=["Tenor", alt.Tooltip("Yield:Q", format=".2f")]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"Latest PKRV date: {max_date.date()}")

with right:
    st.subheader("PKRV History (Selected Tenors)")
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
            hchart = alt.Chart(hist).mark_line().encode(
                x="Date:T",
                y="Yield:Q",
                color="Tenor:N",
                tooltip=["Date:T", "Tenor:N", alt.Tooltip("Yield:Q", format=".2f")]
            ).properties(height=320)
            st.altair_chart(hchart, use_container_width=True)

st.divider()

# ============================
# COST CENTERS TABLE + BASIC PIVOT
# ============================
st.subheader("Cost Centers")

if df_cost is None or df_cost.empty:
    st.warning(
        f"Sheet '{SHEET_COST_CENTERS}' not found or empty. "
        "Expected columns: Month, CostCenter, Amount."
    )
else:
    df_cost2 = df_cost.copy()

    # Basic normalization
    if "Month" in df_cost2.columns:
        df_cost2["Month"] = pd.to_datetime(df_cost2["Month"], errors="coerce")
    if "Amount" in df_cost2.columns:
        df_cost2["Amount"] = pd.to_numeric(df_cost2["Amount"], errors="coerce")

    st.dataframe(df_cost2, use_container_width=True, height=320)

    # Optional summary view if columns exist
    if {"Month", "CostCenter", "Amount"}.issubset(df_cost2.columns):
        st.subheader("Cost Center Summary (Monthly)")
        pivot = (
            df_cost2.dropna(subset=["Month", "CostCenter", "Amount"])
                    .groupby([pd.Grouper(key="Month", freq="M"), "CostCenter"], as_index=False)["Amount"].sum()
        )
        pchart = alt.Chart(pivot).mark_bar().encode(
            x=alt.X("Month:T", title="Month"),
            y=alt.Y("Amount:Q", title="PKR"),
            color="CostCenter:N",
            tooltip=["Month:T", "CostCenter:N", alt.Tooltip("Amount:Q", format=",.0f")]
        ).properties(height=360)
        st.altair_chart(pchart, use_container_width=True)
    else:
        st.caption("To enable cost-center charts, ensure CostCenters has Month, CostCenter, Amount columns.")

st.divider()

# ============================
# OPTIONAL: BENCHMARK / WAM SHEETS (DISPLAY)
# ============================
with st.expander("Optional Sheets (Benchmark / WAM)"):
    bcol, wcol = st.columns(2)
    with bcol:
        st.markdown(f"**{SHEET_BENCH}**")
        st.dataframe(df_bench if df_bench is not None else pd.DataFrame(), use_container_width=True, height=240)
    with wcol:
        st.markdown(f"**{SHEET_WAM}**")
        st.dataframe(df_wam if df_wam is not None else pd.DataFrame(), use_container_width=True, height=240)

st.caption(
    "Note: Ensure the Google Sheet is shared with the service account email in your Streamlit secrets "
    "(st.secrets['gdrive_service_account']['client_email'])."
)
