import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from html import escape
from io import BytesIO

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception:
    ColumnTransformer = None
    RandomForestRegressor = None
    SimpleImputer = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None
    train_test_split = None
    Pipeline = None
    OneHotEncoder = None

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart Production Planning",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# SESSION STATE
# =========================================================

PAGES = [
    "Home",
    "Upload Data",
    "Analysis",
    "Prediction",
    "Scenarios",
    "Decision Support",
    "KPIs Dashboard",
]

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "df" not in st.session_state:
    st.session_state.df = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "notice" not in st.session_state:
    st.session_state.notice = None

if "last_file_signature" not in st.session_state:
    st.session_state.last_file_signature = None

if "filters" not in st.session_state:
    st.session_state.filters = {}

if "kpi_category" not in st.session_state:
    st.session_state.kpi_category = "All"


EXPECTED_DATA_COLUMNS = {
    "date",
    "machine_id",
    "product_id",
    "shift",
    "operator_exp_years",
    "temperature_c",
    "humidity_pct",
    "material_batch",
    "demand_priority",
    "line_speed_units_hr",
    "downtime_min",
    "changeover_min",
    "planned_qty",
    "capacity_available",
    "production_qty",
    "defect_qty",
    "rework_qty",
    "rework_rate",
    "good_qty",
    "energy_kwh",
}

NUMERIC_HINT_COLUMNS = [
    "operator_exp_years",
    "temperature_c",
    "humidity_pct",
    "line_speed_units_hr",
    "downtime_min",
    "changeover_min",
    "planned_qty",
    "capacity_available",
    "production_qty",
    "defect_qty",
    "rework_qty",
    "rework_rate",
    "good_qty",
    "energy_kwh",
]

REQUIRED_PLANNING_COLUMNS = ["planned_qty", "capacity_available", "rework_rate"]
OPTIONAL_FILTER_COLUMNS = ["machine_id", "product_id", "shift", "demand_priority"]
MODEL_FEATURE_COLUMNS = [
    "machine_id",
    "product_id",
    "shift",
    "operator_exp_years",
    "temperature_c",
    "humidity_pct",
    "material_batch",
    "demand_priority",
    "line_speed_units_hr",
    "downtime_min",
    "changeover_min",
    "planned_qty",
    "capacity_available",
    "energy_kwh",
]

FILTER_RENDER_COUNTS = {}
FILTER_PANEL_RENDERED = False
PAGE_FILTERS_RENDERED = set()
PAGE_FILTER_RESULTS = {}
PAGE_BODIES_RENDERED = set()

# Read page from URL if user clicks top navigation
try:
    page_from_url = st.query_params.get("page", None)
    if isinstance(page_from_url, list):
        page_from_url = page_from_url[0]
    if page_from_url in PAGES:
        st.session_state.page = page_from_url
except Exception:
    pass


def go_to(page_name):
    st.session_state.page = page_name
    try:
        st.query_params["page"] = page_name
    except Exception:
        pass
    st.rerun()


# =========================================================
# CSS DESIGN
# =========================================================

st.markdown(
    """
    <style>
    /* ===== GLOBAL ===== */
    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(61, 139, 255, 0.08), transparent 30%),
            radial-gradient(circle at top right, rgba(20, 184, 166, 0.06), transparent 25%),
            #fbfcff;
        color: #0f172a;
    }

    .block-container {
        padding-top: 1.2rem;
        max-width: 1180px;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* ===== NAVBAR ===== */
    .nav-row {
        display: flex;
        align-items: center;
        background: rgba(245, 246, 248, 0.92);
        border: 1px solid rgba(15, 23, 42, 0.06);
        border-radius: 999px;
        padding: 8px 16px;
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.06);
        backdrop-filter: blur(14px);
        margin-bottom: 2.5rem;
        gap: 4px;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 900;
        color: #0b1220;
        font-size: 21px;
        letter-spacing: -0.5px;
        padding-left: 8px;
        white-space: nowrap;
        flex-shrink: 0;
        margin-right: 12px;
    }

    .brand-badge {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0b1220, #1d4ed8);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 16px;
        font-weight: 900;
    }

    /* Nav buttons override */
    .nav-row div.stButton > button {
        border-radius: 999px !important;
        padding: 8px 14px !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        white-space: nowrap;
    }

    .nav-row div.stButton > button:hover {
        background: white !important;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08) !important;
        transform: translateY(-1px) !important;
    }

    .nav-row button[kind="primary"] {
        background: #0b1220 !important;
        color: white !important;
    }

    .nav-row button[kind="secondary"] {
        background: transparent !important;
        color: #111827 !important;
    }

    /* ===== HERO ===== */
    .hero {
        text-align: center;
        max-width: 980px;
        margin: 0 auto;
        padding: 0 10px 25px 10px;
    }

    .hero-logo {
        width: 96px;
        height: 96px;
        border-radius: 28px;
        margin: 0 auto 28px auto;
        background:
            linear-gradient(135deg, rgba(219, 234, 254, 1), rgba(220, 252, 231, 1));
        box-shadow: 0 20px 50px rgba(37, 99, 235, 0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }

    .hero-logo-inner {
        width: 66px;
        height: 66px;
        border-radius: 20px;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 30px;
        font-weight: 950;
        letter-spacing: -1px;
    }

    .hero-title {
        color: #09090b;
        font-size: clamp(54px, 8vw, 92px);
        line-height: 0.98;
        font-weight: 950;
        letter-spacing: -5px;
        margin-bottom: 25px;
    }

    .hero-subtitle {
        color: #737373;
        font-size: clamp(19px, 2vw, 25px);
        line-height: 1.35;
        font-weight: 500;
        max-width: 760px;
        margin: 0 auto 28px auto;
    }

    .trusted {
        text-align: center;
        margin-top: 70px;
        color: #858585;
        font-size: 15px;
        font-weight: 600;
    }

    .logo-row {
        margin-top: 26px;
        display: flex;
        justify-content: center;
        gap: 42px;
        flex-wrap: wrap;
        color: #b6b6b6;
        font-size: 22px;
        font-weight: 800;
    }

    /* ===== CARDS ===== */
    .section-title {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: -2px;
        color: #0b1220;
        margin-bottom: 10px;
    }

    .section-subtitle {
        color: #6b7280;
        font-size: 18px;
        margin-bottom: 28px;
        max-width: 760px;
    }

    .card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 28px;
        padding: 28px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
        margin-bottom: 22px;
    }

    .mini-card {
        background: white;
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 16px 35px rgba(15, 23, 42, 0.05);
        height: 100%;
    }

    .mini-card h3 {
        color: #0b1220;
        font-size: 21px;
        margin-bottom: 8px;
    }

    .mini-card p {
        color: #6b7280;
        font-size: 15px;
        line-height: 1.55;
    }

    .kpi {
        background: white;
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 24px;
        padding: 22px;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.05);
    }

    .kpi-label {
        color: #6b7280;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .kpi-value {
        color: #0b1220;
        font-size: 34px;
        font-weight: 900;
        letter-spacing: -1px;
    }

    .best-card {
        background:
            radial-gradient(circle at top right, rgba(255,255,255,0.25), transparent 40%),
            linear-gradient(135deg, #0b1220, #1d4ed8);
        color: white;
        border-radius: 30px;
        padding: 34px;
        box-shadow: 0 22px 55px rgba(37, 99, 235, 0.25);
        margin: 20px 0;
    }

    .best-card h2 {
        color: white;
        font-size: 34px;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }

    .best-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(120px, 1fr));
        gap: 14px;
        margin-top: 22px;
    }

    .best-item {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 18px;
        padding: 16px;
    }

    .best-item span {
        display: block;
        opacity: 0.75;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .best-item strong {
        display: block;
        font-size: 24px;
    }

    .clean-warning {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        padding: 22px;
        border-radius: 22px;
        font-weight: 650;
        margin-bottom: 18px;
    }

    .status-strip {
        display: grid;
        grid-template-columns: repeat(5, minmax(120px, 1fr));
        gap: 10px;
        margin: 0 0 2rem 0;
    }

    .status-step {
        background: white;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 16px;
        padding: 13px 14px;
        color: #64748b;
        font-size: 13px;
        font-weight: 800;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
    }

    .status-step.done {
        background: #ecfdf5;
        border-color: #86efac;
        color: #166534;
    }

    .quality-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(150px, 1fr));
        gap: 12px;
        margin-top: 18px;
    }

    .quality-item {
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: #ffffff;
        padding: 16px;
    }

    .quality-item strong {
        display: block;
        color: #0b1220;
        font-size: 22px;
        margin-top: 6px;
    }

    .explain-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        color: #334155;
        font-weight: 650;
        margin-bottom: 18px;
    }

    /* ===== STREAMLIT BUTTONS ===== */
    div.stButton > button {
        border-radius: 999px !important;
        padding: 0.75rem 1.4rem !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08) !important;
        white-space: nowrap !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.13) !important;
    }

    button[kind="primary"] {
        background: #1d4ed8 !important;
        color: white !important;
        border-color: #1d4ed8 !important;
    }

    button[kind="secondary"] {
        background: white !important;
        color: #0b1220 !important;
    }

    /* Nav buttons — smaller to fit in one line */
    [data-testid="stHorizontalBlock"]:first-of-type div.stButton > button {
        font-size: 13px !important;
        padding: 7px 11px !important;
        font-weight: 750 !important;
    }

    /* ===== INPUTS ===== */
    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #93c5fd;
        border-radius: 26px;
        padding: 22px;
    }

    /* ===== TABLES ===== */
    [data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 800px) {
        .nav {
            border-radius: 28px;
            flex-direction: column;
            gap: 14px;
        }

        .nav-links {
            justify-content: center;
        }

        .hero-title {
            letter-spacing: -2px;
        }

        .best-grid {
            grid-template-columns: 1fr 1fr;
        }

        .status-strip,
        .quality-grid {
            grid-template-columns: 1fr;
            margin-top: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# NAVIGATION BAR
# =========================================================

def render_navbar():
    active = st.session_state.page

    cols = st.columns([2.0, 0.8, 1.1, 1.0, 1.1, 1.1, 1.5, 1.5])

    with cols[0]:
        st.markdown(
            '<div class="brand"><div class="brand-badge">SP</div><div>SmartPlan</div></div>',
            unsafe_allow_html=True,
        )

    nav_items = [
        ("Home", cols[1]),
        ("Upload Data", cols[2]),
        ("Analysis", cols[3]),
        ("Prediction", cols[4]),
        ("Scenarios", cols[5]),
        ("Decision Support", cols[6]),
        ("KPIs Dashboard", cols[7]),
    ]

    for name, col in nav_items:
        with col:
            if st.button(
                name,
                key=f"nav_{name.lower().replace(' ', '_')}_btn",
                use_container_width=True,
                type="primary" if active == name else "secondary",
            ):
                go_to(name)

    st.markdown('<hr style="border:none;border-top:1px solid #e5e7eb;margin:0.4rem 0 2.5rem 0;">', unsafe_allow_html=True)


render_navbar()


def render_status_strip():
    states = [
        ("Data", st.session_state.df is not None),
        ("Analysis", st.session_state.df is not None),
        ("Prediction", "df_pred" in st.session_state),
        ("Scenarios", "scenarios_df" in st.session_state),
        ("Decision", "results_df" in st.session_state),
    ]
    html = "".join(
        f'<div class="status-step {"done" if done else ""}">{escape(label)}</div>'
        for label, done in states
    )
    st.markdown(f'<div class="status-strip">{html}</div>', unsafe_allow_html=True)


render_status_strip()


# =========================================================
# HELPERS
# =========================================================

def get_df():
    return st.session_state.df


def enter_page_once(page_key):
    if page_key in PAGE_BODIES_RENDERED:
        return False
    PAGE_BODIES_RENDERED.add(page_key)
    return True


def require_data():
    if st.session_state.df is None:
        st.markdown(
            """
            <div class="clean-warning">
                Please upload a CSV or Excel dataset first.
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Go to Upload Data", type="primary", key="require_data_go_upload_button"):
            go_to("Upload Data")
        return False
    return True


def style_plot(ax):
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    ax.tick_params(colors="#111827", labelsize=10)
    ax.xaxis.label.set_color("#111827")
    ax.yaxis.label.set_color("#111827")
    ax.title.set_color("#0b1220")
    ax.title.set_fontweight("bold")
    ax.grid(True, alpha=0.22, color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#e5e7eb")


def kpi_card(label, value):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{escape(str(label))}</div>
            <div class="kpi-value">{escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(title, subtitle):
    st.markdown(f"<div class='section-title'>{escape(str(title))}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-subtitle'>{escape(str(subtitle))}</div>", unsafe_allow_html=True)


def reset_derived_state():
    for key in ("df_pred", "scenarios_df", "results_df"):
        st.session_state.pop(key, None)


def render_filters():
    # Deprecated: page sections now use render_page_filters(), which owns
    # page-scoped widget keys and avoids StreamlitDuplicateElementId errors.
    return None


def render_page_filters(df, scope):
    """Render one safely-keyed filter panel for the active page and return filtered rows."""
    global PAGE_FILTERS_RENDERED, PAGE_FILTER_RESULTS

    if df is None or df.empty:
        return df

    if scope in PAGE_FILTERS_RENDERED:
        return PAGE_FILTER_RESULTS.get(scope, df).copy()

    PAGE_FILTERS_RENDERED.add(scope)
    result = df.copy()

    with st.expander("Production filters", expanded=False):
        filters_used = False
        cols = st.columns(4)

        for i, col in enumerate(OPTIONAL_FILTER_COLUMNS):
            if col not in result.columns:
                continue

            options = sorted(result[col].dropna().astype(str).unique().tolist())
            if not options:
                continue

            selected = cols[i % 4].multiselect(
                col.replace("_", " ").title(),
                options,
                key=f"{scope}_{col}_filter_multiselect",
            )
            filters_used = True

            if selected:
                result = result[result[col].astype(str).isin(selected)]

        if "date" in result.columns:
            dates = pd.to_datetime(result["date"], errors="coerce").dropna()
            if not dates.empty:
                date_range = st.date_input(
                    "Date range",
                    value=(dates.min().date(), dates.max().date()),
                    key=f"{scope}_date_filter_input",
                )
                filters_used = True

                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    all_dates = pd.to_datetime(result["date"], errors="coerce")
                    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                    result = result[(all_dates >= start) & (all_dates <= end)]

        if not filters_used:
            st.caption("No categorical or date filters are available for this dataset.")

    result = result.reset_index(drop=True)
    PAGE_FILTER_RESULTS[scope] = result
    return result


def validation_report(df):
    missing_required = [col for col in REQUIRED_PLANNING_COLUMNS if col not in df.columns]
    present_optional = [col for col in OPTIONAL_FILTER_COLUMNS if col in df.columns]
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    numeric_count = len(df.select_dtypes(include=np.number).columns)
    quality_score = 100
    quality_score -= len(missing_required) * 18
    quality_score -= min(25, int((missing_values / max(1, df.size)) * 100))
    quality_score -= min(15, duplicate_rows)
    quality_score = max(0, quality_score)

    return {
        "missing_required": missing_required,
        "present_optional": present_optional,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_count": numeric_count,
        "quality_score": quality_score,
    }


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(sheets):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()


def clean_loaded_dataframe(df):
    """Normalize uploaded project datasets, including files with metadata rows."""
    if df is None or df.empty:
        raise ValueError("The uploaded file is empty.")

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
    if df.empty:
        raise ValueError("The uploaded file contains no usable rows or columns.")

    normalized_columns = {str(col).strip().lower() for col in df.columns}
    if len(normalized_columns.intersection(EXPECTED_DATA_COLUMNS)) < 3:
        header_row_index = None
        for idx, row in df.head(15).iterrows():
            row_values = [str(value).strip().lower() for value in row.tolist() if pd.notna(value)]
            if len(set(row_values).intersection(EXPECTED_DATA_COLUMNS)) >= 5:
                header_row_index = idx
                break

        if header_row_index is not None:
            new_columns = df.loc[header_row_index].tolist()
            df = df.loc[header_row_index + 1:].copy()
            df.columns = [str(col).strip() for col in new_columns]
            df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    df.columns = [str(col).strip() for col in df.columns]
    valid_columns = [
        bool(col) and str(col).lower() != "nan" and not str(col).lower().startswith("unnamed")
        for col in df.columns
    ]
    df = df.loc[:, valid_columns]

    if df.empty or len(df.columns) == 0:
        raise ValueError("No usable columns were found after cleaning the uploaded file.")

    for col in NUMERIC_HINT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def load_uploaded_dataframe(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file, sep=None, engine="python")
        else:
            raw_df = pd.read_excel(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        raw_df = pd.read_csv(uploaded_file, sep=None, engine="python", encoding="latin1")

    return clean_loaded_dataframe(raw_df)


def positive_number(value, fallback=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def close_plot(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plotly_available():
    return px is not None and go is not None


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 0.0 if denominator == 0 else float(1 - np.sum((y_true - y_pred) ** 2) / denominator)
    return mae, rmse, r2


def train_numpy_rework_model(df):
    if "rework_rate" not in df.columns:
        raise ValueError("The target column 'rework_rate' is required for model training.")

    feature_cols = [col for col in MODEL_FEATURE_COLUMNS if col in df.columns]
    if not feature_cols:
        raise ValueError("No usable model feature columns were found.")

    model_df = df[feature_cols + ["rework_rate"]].copy()
    model_df["rework_rate"] = pd.to_numeric(model_df["rework_rate"], errors="coerce")
    model_df = model_df.dropna(subset=["rework_rate"])

    if len(model_df) < 20:
        raise ValueError("At least 20 rows with a valid rework_rate are required to train the model.")

    X_raw = model_df[feature_cols].copy()
    for col in X_raw.columns:
        if col in NUMERIC_HINT_COLUMNS:
            X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    X_encoded = pd.get_dummies(X_raw, dummy_na=True)
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True)).fillna(0)

    y = model_df["rework_rate"].astype(float).to_numpy()
    rng = np.random.default_rng(42)
    indices = np.arange(len(X_encoded))
    rng.shuffle(indices)
    test_size = max(1, int(len(indices) * 0.2))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    X_train = X_encoded.iloc[train_idx].to_numpy(dtype=float)
    X_test = X_encoded.iloc[test_idx].to_numpy(dtype=float)
    y_train = y[train_idx]
    y_test = y[test_idx]

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1

    X_train_scaled = (X_train - means) / stds
    X_test_scaled = (X_test - means) / stds
    X_train_design = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
    X_test_design = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

    alpha = 0.2
    identity = np.eye(X_train_design.shape[1])
    identity[0, 0] = 0
    coefficients = np.linalg.pinv(X_train_design.T @ X_train_design + alpha * identity) @ X_train_design.T @ y_train
    y_pred = np.clip(X_test_design @ coefficients, 0, 1)
    mae, rmse, r2 = regression_metrics(y_test, y_pred)

    full_raw = df.reindex(columns=feature_cols).copy()
    for col in full_raw.columns:
        if col in NUMERIC_HINT_COLUMNS:
            full_raw[col] = pd.to_numeric(full_raw[col], errors="coerce")

    full_encoded = pd.get_dummies(full_raw, dummy_na=True)
    full_encoded = full_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    full_encoded = full_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
    full_scaled = (full_encoded.to_numpy(dtype=float) - means) / stds
    full_design = np.column_stack([np.ones(len(full_scaled)), full_scaled])

    df_pred = df.copy()
    df_pred["predicted_rework_rate"] = np.clip(full_design @ coefficients, 0, 1)

    if {"defect_qty", "production_qty"}.issubset(df_pred.columns):
        production_qty = pd.to_numeric(df_pred["production_qty"], errors="coerce").replace(0, np.nan)
        defect_qty = pd.to_numeric(df_pred["defect_qty"], errors="coerce")
        df_pred["predicted_defect_rate"] = np.clip(
            (defect_qty / production_qty).fillna(df_pred["predicted_rework_rate"] * 1.2),
            0,
            1,
        )
    else:
        df_pred["predicted_defect_rate"] = np.clip(df_pred["predicted_rework_rate"] * 1.2, 0, 1)

    importances = pd.DataFrame(
        {
            "feature": X_encoded.columns,
            "importance": np.abs(coefficients[1:]),
        }
    ).sort_values("importance", ascending=False).head(15)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Rows": len(model_df),
        "Features": len(feature_cols),
        "Model": "NumPy ridge regression",
    }

    return df_pred, metrics, importances


def train_rework_model(df):
    if RandomForestRegressor is None:
        return train_numpy_rework_model(df)

    if "rework_rate" not in df.columns:
        raise ValueError("The target column 'rework_rate' is required for model training.")

    model_df = df.dropna(subset=["rework_rate"]).copy()
    feature_cols = [col for col in MODEL_FEATURE_COLUMNS if col in model_df.columns]

    if len(model_df) < 20:
        raise ValueError("At least 20 usable rows are required to train the prediction model.")
    if not feature_cols:
        raise ValueError("No usable model feature columns were found.")

    X = model_df[feature_cols]
    y = pd.to_numeric(model_df["rework_rate"], errors="coerce")
    valid_target = y.notna()
    X = X.loc[valid_target]
    y = y.loc[valid_target]

    if len(X) < 20:
        raise ValueError("At least 20 rows with a valid rework_rate are required to train the model.")

    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    transformers = []
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                num_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)
    if XGBRegressor is not None:
        estimator = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
        )
        model_name = "XGBoost"
    else:
        estimator = RandomForestRegressor(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        model_name = "Random Forest"

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 0, 1)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": r2_score(y_test, y_pred),
        "Rows": len(X),
        "Features": len(feature_cols),
        "Model": model_name,
    }

    full_pred = df.copy()
    full_X = full_pred.reindex(columns=feature_cols)
    full_pred["predicted_rework_rate"] = np.clip(model.predict(full_X), 0, 1)

    if {"defect_qty", "production_qty"}.issubset(full_pred.columns):
        production_qty = pd.to_numeric(full_pred["production_qty"], errors="coerce").replace(0, np.nan)
        defect_qty = pd.to_numeric(full_pred["defect_qty"], errors="coerce")
        full_pred["predicted_defect_rate"] = np.clip(
            (defect_qty / production_qty).fillna(full_pred["predicted_rework_rate"] * 1.2),
            0,
            1,
        )
    else:
        full_pred["predicted_defect_rate"] = np.clip(full_pred["predicted_rework_rate"] * 1.2, 0, 1)

    importances = pd.DataFrame()
    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
        raw_importances = model.named_steps["model"].feature_importances_
        importances = pd.DataFrame({"feature": names, "importance": raw_importances})
        importances = importances.sort_values("importance", ascending=False).head(15)
    except Exception:
        importances = pd.DataFrame()

    return full_pred, metrics, importances


def build_prediction_dataframe(df):
    df_pred = df.copy()
    rng = np.random.default_rng(42)

    if "rework_rate" in df_pred.columns and df_pred["rework_rate"].notna().any():
        base_rework = pd.to_numeric(df_pred["rework_rate"], errors="coerce")
        base_rework = base_rework.fillna(base_rework.median())
        noise = rng.normal(loc=1.0, scale=0.06, size=len(df_pred))
        df_pred["predicted_rework_rate"] = np.clip(base_rework * noise, 0, 1)
    else:
        df_pred["predicted_rework_rate"] = rng.uniform(0.015, 0.075, size=len(df_pred))

    if {"defect_qty", "production_qty"}.issubset(df_pred.columns):
        production_qty = pd.to_numeric(df_pred["production_qty"], errors="coerce").replace(0, np.nan)
        defect_qty = pd.to_numeric(df_pred["defect_qty"], errors="coerce")
        base_defect = (defect_qty / production_qty).fillna(df_pred["predicted_rework_rate"])
        df_pred["predicted_defect_rate"] = np.clip(base_defect, 0, 1)
    else:
        multiplier = rng.uniform(1.08, 1.35, size=len(df_pred))
        df_pred["predicted_defect_rate"] = np.clip(df_pred["predicted_rework_rate"] * multiplier, 0, 1)

    return df_pred


# =========================================================
# HOME PAGE
# =========================================================

def render_home():
    if not enter_page_once("home"):
        return

    st.markdown(
        """
        <div class="hero">
            <div class="hero-logo">
                <div class="hero-logo-inner">SP</div>
            </div>
            <div class="hero-title">
                Smart production planning<br>
                under uncertainty.
            </div>
            <div class="hero-subtitle">
                Upload your production dataset, predict defect and rework behavior,
                generate stochastic scenarios, and select the best plan with decision support.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    left, mid1, mid2, right = st.columns([2.2, 1.15, 1.15, 2.2])

    with mid1:
        if st.button("Upload Dataset", type="primary", use_container_width=True, key="home_upload_button"):
            go_to("Upload Data")

    with mid2:
        if st.button("Start Work →", type="secondary", use_container_width=True, key="home_start_work_button"):
            if st.session_state.df is None:
                st.session_state.notice = "Please upload your dataset first before starting the work."
                go_to("Upload Data")
            else:
                go_to("Analysis")

    st.markdown(
        """
        <div class="trusted">
            Project workflow
            <div class="logo-row">
                <span>Data</span>
                <span>ML Prediction</span>
                <span>Scenarios</span>
                <span>Optimization</span>
                <span>Decision</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")
    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="mini-card">
                <h3>📂 Upload & inspect</h3>
                <p>Load CSV or Excel datasets and instantly understand rows, columns, missing values and structure.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="mini-card">
                <h3>🤖 Predict quality risk</h3>
                <p>Simulate defect and rework rates now, then replace this module later with your XGBoost model.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="mini-card">
                <h3>🏆 Choose best scenario</h3>
                <p>Compare production scenarios using satisfaction, shortage, capacity utilization, stability and cost.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# UPLOAD PAGE
# =========================================================

def render_upload():
    if not enter_page_once("upload"):
        return

    page_header(
        "Upload Data",
        "Import your production dataset and inspect its structure before running the analysis."
    )

    if st.session_state.notice:
        st.markdown(
            f"""
            <div class="clean-warning">
                {escape(str(st.session_state.notice))}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.session_state.notice = None

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx"],
        key="upload_data_file_uploader",
    )

    if uploaded_file is not None:
        try:
            file_signature = (uploaded_file.name, uploaded_file.size)
            df = load_uploaded_dataframe(uploaded_file)

            if file_signature != st.session_state.last_file_signature:
                reset_derived_state()
                st.session_state.last_file_signature = file_signature

            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name

            st.success("Dataset uploaded successfully.")

        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as e:
            st.error(f"Could not read the file: {e}")
        except Exception as e:
            st.error(f"Unexpected loading error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    df = get_df()

    if df is not None:
        st.write("")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            kpi_card("File", st.session_state.file_name)
        with c2:
            kpi_card("Rows", f"{df.shape[0]:,}")
        with c3:
            kpi_card("Columns", df.shape[1])
        with c4:
            kpi_card("Missing", int(df.isna().sum().sum()))

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Column list")
        st.write(list(df.columns))
        st.markdown("</div>", unsafe_allow_html=True)

        report = validation_report(df)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset validation")
        st.markdown(
            f"""
            <div class="quality-grid">
                <div class="quality-item">Quality score<strong>{report["quality_score"]}/100</strong></div>
                <div class="quality-item">Numeric columns<strong>{report["numeric_count"]}</strong></div>
                <div class="quality-item">Duplicate rows<strong>{report["duplicate_rows"]}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if report["missing_required"]:
            st.warning(
                "Missing planning columns: "
                + ", ".join(report["missing_required"])
                + ". The app will use safe fallback assumptions where needed."
            )
        else:
            st.success("All core planning columns are available.")
        if report["present_optional"]:
            st.info("Available filter dimensions: " + ", ".join(report["present_optional"]))
        st.download_button(
            "Download cleaned dataset",
            data=to_csv_bytes(df),
            file_name="cleaned_production_dataset.csv",
            mime="text/csv",
            type="secondary",
            key="upload_cleaned_dataset_download",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Continue to Analysis →", type="primary", key="upload_continue_analysis_button"):
            go_to("Analysis")


# =========================================================
# ANALYSIS PAGE
# =========================================================

def render_analysis():
    if not enter_page_once("analysis"):
        return

    page_header(
        "Analysis",
        "Explore your numeric variables with clean indicators, readable charts, and descriptive statistics."
    )

    if not require_data():
        return

    df = render_page_filters(get_df(), "analysis")
    if df is None or df.empty:
        st.warning("No rows match the current filters.")
        return
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) == 0:
        st.markdown(
            """
            <div class="clean-warning">
                No numeric columns were found in this dataset.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    selected_col = st.selectbox(
        "Choose a numeric column",
        numeric_cols,
        key="analysis_numeric_column_selectbox",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Mean", round(df[selected_col].mean(), 3))
    with c2:
        kpi_card("Minimum", round(df[selected_col].min(), 3))
    with c3:
        kpi_card("Maximum", round(df[selected_col].max(), 3))
    with c4:
        kpi_card("Std. Dev", round(df[selected_col].std(), 3))

    st.download_button(
        "Download filtered data",
        data=to_csv_bytes(df),
        file_name="filtered_production_data.csv",
        mime="text/csv",
        type="secondary",
        key="analysis_filtered_data_download",
    )

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(f"Distribution of {selected_col}")
        if plotly_available():
            fig = px.histogram(
                df,
                x=selected_col,
                nbins=24,
                color_discrete_sequence=["#2563eb"],
                template="plotly_white",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.hist(df[selected_col].dropna(), bins=22, color="#60a5fa", edgecolor="white", linewidth=1.2)
            ax.set_xlabel(selected_col)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram of {selected_col}")
            style_plot(ax)
            close_plot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Descriptive statistics")
        st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if {"planned_qty", "capacity_available"}.issubset(df.columns) and plotly_available():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Planned quantity vs available capacity")
        fig = px.scatter(
            df,
            x="planned_qty",
            y="capacity_available",
            color="machine_id" if "machine_id" in df.columns else None,
            hover_data=[col for col in ["product_id", "shift", "rework_rate"] if col in df.columns],
            template="plotly_white",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Continue to Prediction →", type="primary", key="analysis_continue_prediction_button"):
        go_to("Prediction")


# =========================================================
# PREDICTION PAGE
# =========================================================

def render_prediction():
    if not enter_page_once("prediction"):
        return

    page_header(
        "Prediction",
        "Simulate predicted defect and rework rates. This module can later be replaced by your trained XGBoost model."
    )

    if not require_data():
        return

    df = render_page_filters(get_df(), "prediction")
    if df is None or df.empty:
        st.warning("No rows match the current filters.")
        return

    st.markdown(
        """
        <div class="card">
            <b>Prediction mode:</b> trained model when the target column <code>rework_rate</code> is available;
            otherwise the app falls back to a transparent simulation based on production quantities.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Run Prediction", type="primary", key="prediction_run_button"):
        try:
            df_pred, model_metrics, importances = train_rework_model(df)
            st.session_state.model_metrics = model_metrics
            st.session_state.importances = importances
            st.success("Prediction completed with the trained model.")
        except Exception as e:
            df_pred = build_prediction_dataframe(df)
            st.session_state.model_metrics = None
            st.session_state.importances = pd.DataFrame()
            st.warning(f"Model training fallback used: {e}")

        st.session_state.df_pred = df_pred

    if "df_pred" in st.session_state:
        df_pred = st.session_state.df_pred

        c1, c2, c3 = st.columns(3)

        with c1:
            kpi_card("Avg. defect rate", f"{df_pred['predicted_defect_rate'].mean():.2%}")
        with c2:
            kpi_card("Avg. rework rate", f"{df_pred['predicted_rework_rate'].mean():.2%}")
        with c3:
            kpi_card("Predicted rows", f"{len(df_pred):,}")

        if st.session_state.get("model_metrics"):
            metrics = st.session_state.model_metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                kpi_card("Model MAE", f"{metrics['MAE']:.4f}")
            with m2:
                kpi_card("Model RMSE", f"{metrics['RMSE']:.4f}")
            with m3:
                kpi_card("Model R2", f"{metrics['R2']:.3f}")
            kpi_card("Model used", metrics.get("Model", "Model"))

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Prediction preview")
        st.dataframe(df_pred.head(20), use_container_width=True)
        st.download_button(
            "Download predictions",
            data=to_csv_bytes(df_pred),
            file_name="production_rework_predictions.csv",
            mime="text/csv",
            type="secondary",
            key="prediction_download_button",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Predicted defect rate distribution")
        if plotly_available():
            fig = px.histogram(
                df_pred,
                x="predicted_defect_rate",
                nbins=24,
                color_discrete_sequence=["#14b8a6"],
                template="plotly_white",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.hist(df_pred["predicted_defect_rate"], bins=22, color="#5eead4", edgecolor="white", linewidth=1.2)
            ax.set_xlabel("Predicted defect rate")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of predicted defect rate")
            style_plot(ax)
            close_plot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        importances = st.session_state.get("importances", pd.DataFrame())
        if isinstance(importances, pd.DataFrame) and not importances.empty and plotly_available():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Model feature importance")
            fig = px.bar(
                importances.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                template="plotly_white",
                color_discrete_sequence=["#1d4ed8"],
            )
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if "df_pred" in st.session_state:
        if st.button("Continue to Scenarios →", type="primary", key="prediction_continue_scenarios_button"):
            go_to("Scenarios")


# =========================================================
# SCENARIOS PAGE
# =========================================================

def render_scenarios():
    if not enter_page_once("scenarios"):
        return

    page_header(
        "Scenarios",
        "Generate stochastic scenarios by varying demand, production capacity and defect rate."
    )

    if not require_data():
        return

    df = get_df()
    default_capacity = 163
    if "capacity_available" in df.columns and df["capacity_available"].notna().any():
        default_capacity = int(round(positive_number(df["capacity_available"].median(), 163)))

    default_energy = 2.5
    if "energy_kwh" in df.columns and df["energy_kwh"].notna().any():
        total_kwh = pd.to_numeric(df["energy_kwh"], errors="coerce").dropna()
        prod_qty = pd.to_numeric(df.get("production_qty", pd.Series(dtype=float)), errors="coerce").dropna()
        if not total_kwh.empty and not prod_qty.empty and prod_qty.mean() > 0:
            default_energy = round(float(total_kwh.mean() / prod_qty.mean()), 4)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        n_scenarios = st.number_input(
            "Number of scenarios",
            min_value=1,
            max_value=20,
            value=3,
            key="scenarios_count_input",
        )

    with c2:
        capacity_mean = st.number_input(
            "Average capacity per shift",
            min_value=1,
            value=default_capacity,
            key="scenarios_capacity_mean_input",
        )

    with c3:
        capacity_std = st.number_input(
            "Capacity standard deviation",
            min_value=0,
            value=10,
            key="scenarios_capacity_std_input",
        )

    with c4:
        rework_alpha = st.number_input(
            "Rework efficiency α (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            format="%.2f",
            key="scenarios_rework_alpha_input",
            help="Fraction of defective units successfully reworked into good units.",
        )

    with c5:
        energy_per_unit = st.number_input(
            "Energy per unit (kWh/unit)",
            min_value=0.0,
            value=float(default_energy),
            step=0.1,
            format="%.4f",
            key="scenarios_energy_per_unit_input",
            help="Average energy consumption per produced unit.",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Generate Scenarios", type="primary", key="scenarios_generate_button"):
        scenarios = []
        rng = np.random.default_rng(10)
        demand_source = None
        if "planned_qty" in df.columns and df["planned_qty"].notna().any():
            demand_source = pd.to_numeric(df["planned_qty"], errors="coerce").dropna()

        for s in range(1, int(n_scenarios) + 1):
            if demand_source is not None and not demand_source.empty:
                demand = float(rng.choice(demand_source.to_numpy()))
            else:
                demand = float(rng.integers(120, 180))

            capacity = max(1.0, rng.normal(capacity_mean, capacity_std))
            defect_rate = float(rng.uniform(0.02, 0.08))

            scenarios.append(
                {
                    "Scenario": s,
                    "Demand": demand,
                    "Capacity": round(capacity, 2),
                    "Defect_Rate": round(defect_rate, 4),
                    "Rework_Alpha": round(float(rework_alpha), 4),
                    "Energy_Per_Unit": round(float(energy_per_unit), 4),
                }
            )

        scenarios_df = pd.DataFrame(scenarios)
        st.session_state.scenarios_df = scenarios_df

        st.success("Scenarios generated successfully.")

    if "scenarios_df" in st.session_state:
        scenarios_df = st.session_state.scenarios_df

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Generated scenarios")
        st.dataframe(scenarios_df, use_container_width=True)
        st.download_button(
            "Download scenarios",
            data=to_csv_bytes(scenarios_df),
            file_name="generated_production_scenarios.csv",
            mime="text/csv",
            type="secondary",
            key="scenarios_download_button",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Capacity by scenario")
        if plotly_available():
            fig = px.bar(
                scenarios_df,
                x="Scenario",
                y="Capacity",
                color="Defect_Rate",
                hover_data=["Demand"],
                template="plotly_white",
                color_continuous_scale="Teal",
            )
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.bar(scenarios_df["Scenario"], scenarios_df["Capacity"], color="#93c5fd", edgecolor="#2563eb", linewidth=1)
            ax.set_xlabel("Scenario")
            ax.set_ylabel("Capacity")
            ax.set_title("Available capacity per scenario")
            style_plot(ax)
            close_plot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Continue to Decision Support →", type="primary", key="scenarios_continue_decision_button"):
            go_to("Decision Support")


# =========================================================
# DECISION SUPPORT PAGE
# =========================================================

def render_decision():
    if not enter_page_once("decision"):
        return

    page_header(
        "Decision Support",
        "Compare scenarios using KPI-based multi-criteria scoring (service, capacity, costs, energy)."
    )

    if not require_data():
        return

    if "scenarios_df" not in st.session_state:
        st.markdown(
            """
            <div class="clean-warning">
                Please generate scenarios first before running decision support.
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Go to Scenarios", type="primary", key="decision_go_scenarios_button"):
            go_to("Scenarios")
        return

    # ── KPI constants ──────────────────────────────────────────────────────────
    C_UNIT = 20.0       # €/unit production cost
    C_REWORK = 8.0      # €/unit rework processing cost
    P_SHORT = 1000.0    # €/unit shortage penalty
    P_E = 0.12          # €/kWh energy price
    EF = 0.233          # kg CO2/kWh emission factor
    LAMBDA_CO2 = 0.05   # €/kg CO2 carbon cost
    BETA = 1.0          # rework load factor (load multiplier for defective units)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Multi-criteria weights")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        w_ts = st.slider("Service rate (TS)", 0.0, 1.0, 0.35, key="decision_weight_ts")
    with c2:
        w_uc = st.slider("Capacity use (UC)", 0.0, 1.0, 0.20, key="decision_weight_uc")
    with c3:
        w_stab = st.slider("Stability", 0.0, 1.0, 0.20, key="decision_weight_stability")
    with c4:
        w_tp = st.slider("Shortage penalty (TP)", 0.0, 1.0, 0.25, key="decision_weight_tp")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run Optimization", type="primary", key="decision_run_optimization_button"):
        scenarios_df = st.session_state.scenarios_df
        total_weight = w_ts + w_uc + w_stab + w_tp

        if total_weight <= 0:
            st.error("At least one decision weight must be greater than zero.")
            return

        wn_ts, wn_uc, wn_stab, wn_tp = [
            w / total_weight for w in (w_ts, w_uc, w_stab, w_tp)
        ]

        # ── Pass 1: compute per-scenario KPIs (except cross-scenario Stab) ──
        rows = []
        q_vals = []

        for _, row in scenarios_df.iterrows():
            scenario = int(row["Scenario"])
            demand = positive_number(row["Demand"])
            capacity = positive_number(row["Capacity"])
            r = min(max(float(row["Defect_Rate"]), 0.0), 1.0)
            alpha = min(max(float(row.get("Rework_Alpha", 0.70)), 0.0), 1.0)
            e = max(float(row.get("Energy_Per_Unit", 2.5)), 0.0)

            # Actual production quantity (limited by capacity)
            q = min(demand, capacity)
            q_vals.append(q)

            # Good production: non-defective + successfully reworked fraction
            p_good = q * (1.0 - r) + alpha * q * r  # = q*(1 - r*(1-alpha))

            # Shortage
            shortage = max(0.0, demand - p_good)

            # Service rate: TS = P_good / Demand
            TS = min(p_good / demand, 1.0) if demand > 0 else 1.0

            # Shortage rate: TP = shortage / Demand
            TP = shortage / demand if demand > 0 else 0.0

            # Capacity utilization (including rework load): UC = q*(1+β*r) / capacity
            UC = min(q * (1.0 + BETA * r) / capacity, 1.5) if capacity > 0 else 0.0

            # Rework efficiency: TRA = α (from user input)
            TRA = alpha

            # Costs
            C_prod = q * C_UNIT
            C_rework_cost = alpha * q * r * C_REWORK
            C_penurie = P_SHORT * shortage
            C_total = C_prod + C_rework_cost + C_penurie

            # Unit cost
            CU = C_total / p_good if p_good > 0 else float("nan")

            # Energy & integrated environmental-cost indicator
            E_tot = e * q
            IECP = E_tot * (P_E + LAMBDA_CO2 * EF)

            rows.append({
                "Scenario": scenario,
                "Demand": round(demand, 2),
                "P_Good": round(p_good, 2),
                "Shortage": round(shortage, 2),
                "TS_%": round(TS * 100, 2),
                "TP_%": round(TP * 100, 2),
                "UC_%": round(UC * 100, 2),
                "TRA_%": round(TRA * 100, 2),
                "C_Prod": round(C_prod, 2),
                "C_Rework": round(C_rework_cost, 2),
                "C_Penurie": round(C_penurie, 2),
                "C_Total": round(C_total, 2),
                "CU": round(CU, 2) if not np.isnan(CU) else 0.0,
                "E_Tot_kWh": round(E_tot, 2),
                "IECP_eur": round(IECP, 2),
                "_q": q,
            })

        # ── Pass 2: cross-scenario stability ──────────────────────────────────
        mean_q = float(np.mean(q_vals)) if q_vals else 1.0
        n = len(rows)

        for i, r_dict in enumerate(rows):
            q_i = q_vals[i]
            other_q = [q_vals[j] for j in range(n) if j != i]
            if other_q and mean_q > 0:
                stab = max(
                    0.0,
                    100.0 - sum(abs(q_i - qj) for qj in other_q) / len(other_q) / mean_q * 100.0,
                )
            else:
                stab = 100.0
            r_dict["Stab"] = round(stab, 2)

        results_df = pd.DataFrame(rows).drop(columns=["_q"])

        # ── Score (weighted, normalized) ──────────────────────────────────────
        TS_norm = results_df["TS_%"] / 100.0
        UC_norm = (results_df["UC_%"] / 100.0).clip(0, 1)
        Stab_norm = results_df["Stab"] / 100.0
        TP_norm = results_df["TP_%"] / 100.0

        results_df["Score"] = (
            wn_ts * TS_norm
            + wn_uc * UC_norm
            + wn_stab * Stab_norm
            - wn_tp * TP_norm
        ).round(4)

        st.session_state.results_df = results_df
        st.success("Optimization completed successfully.")

    if "results_df" in st.session_state:
        results_df = st.session_state.results_df
        best = results_df.loc[results_df["Score"].idxmax()]

        explanation = (
            f"Scenario {int(best['Scenario'])} is recommended: "
            f"Service rate {best['TS_%']:.1f}%, "
            f"Capacity use {best['UC_%']:.1f}%, "
            f"Stability {best['Stab']:.1f}, "
            f"Shortage {best['Shortage']:.2f} units, "
            f"Total cost {best['C_Total']:,.0f} €, "
            f"IECP {best['IECP_eur']:,.2f} €."
        )

        st.markdown(
            f"""
            <div class="best-card">
                <h2>🏆 Best Recommended Scenario</h2>
                <p>Selected using weighted multi-criteria score over service, capacity, stability and shortage KPIs.</p>
                <div class="best-grid">
                    <div class="best-item">
                        <span>Scenario</span>
                        <strong>{int(best["Scenario"])}</strong>
                    </div>
                    <div class="best-item">
                        <span>Score</span>
                        <strong>{round(best["Score"], 3)}</strong>
                    </div>
                    <div class="best-item">
                        <span>Service rate (TS)</span>
                        <strong>{round(best["TS_%"], 1)}%</strong>
                    </div>
                    <div class="best-item">
                        <span>Shortage (TP)</span>
                        <strong>{round(best["TP_%"], 1)}%</strong>
                    </div>
                    <div class="best-item">
                        <span>Capacity (UC)</span>
                        <strong>{round(best["UC_%"], 1)}%</strong>
                    </div>
                    <div class="best-item">
                        <span>Stability</span>
                        <strong>{round(best["Stab"], 1)}</strong>
                    </div>
                    <div class="best-item">
                        <span>Total Cost</span>
                        <strong>{round(best["C_Total"], 0):,.0f} €</strong>
                    </div>
                    <div class="best-item">
                        <span>IECP</span>
                        <strong>{round(best["IECP_eur"], 2):,.2f} €</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="explain-box">{escape(explanation)}</div>',
            unsafe_allow_html=True,
        )

        display_cols = [
            "Scenario", "Demand", "P_Good", "Shortage",
            "TS_%", "TP_%", "UC_%", "Stab", "TRA_%",
            "C_Prod", "C_Rework", "C_Penurie", "C_Total", "CU",
            "E_Tot_kWh", "IECP_eur", "Score",
        ]
        display_cols = [c for c in display_cols if c in results_df.columns]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Optimization results")
        st.dataframe(results_df[display_cols], use_container_width=True)
        st.download_button(
            "Download decision report",
            data=to_excel_bytes(
                {
                    "optimization_results": results_df[display_cols],
                    "best_scenario": pd.DataFrame([best[display_cols].to_dict()]),
                }
            ),
            file_name="smart_planning_decision_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="secondary",
            key="decision_report_download_button",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Multi-criteria score by scenario")
        if plotly_available():
            fig = px.bar(
                results_df,
                x="Scenario",
                y="Score",
                color="TP_%",
                hover_data=["TS_%", "UC_%", "Stab", "C_Total", "IECP_eur"],
                template="plotly_white",
                color_continuous_scale="RdYlGn_r",
                labels={"TP_%": "Shortage rate (%)"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(8, 4.6))
            ax.bar(results_df["Scenario"], results_df["Score"], color="#86efac", edgecolor="#16a34a", linewidth=1)
            ax.set_xlabel("Scenario")
            ax.set_ylabel("Score")
            ax.set_title("Scenario ranking based on multi-criteria score")
            style_plot(ax)
            close_plot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        if plotly_available():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Cost vs Service rate trade-off")
            fig = px.scatter(
                results_df,
                x="C_Total",
                y="TS_%",
                size="UC_%",
                color="Score",
                hover_name="Scenario",
                hover_data=["Stab", "IECP_eur", "TP_%"],
                template="plotly_white",
                color_continuous_scale="Blues",
                labels={"C_Total": "Total Cost (€)", "TS_%": "Service Rate (%)"},
            )
            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Best scenario radar")
            categories = ["TS_%", "UC_%", "Stab", "TRA_%", "Score"]
            cat_labels = ["Service (TS)", "Capacity (UC)", "Stability", "Rework (TRA)", "Score"]
            values = [min(max(float(best[c]) / 100.0 if c != "Score" else float(best[c]), 0.0), 1.0) for c in categories]
            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=cat_labels + [cat_labels[0]],
                    fill="toself",
                    name=f"Scenario {int(best['Scenario'])}",
                )
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_white",
                margin=dict(l=20, r=20, t=35, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button("View KPIs Dashboard →", type="primary", key="decision_go_kpi_dashboard_button"):
            go_to("KPIs Dashboard")


# =========================================================
# KPIs DASHBOARD PAGE
# =========================================================

KPI_CATEGORIES = {
    "All": [
        ("TS_%",       "Service Rate TS (%)",          "Service"),
        ("TP_%",       "Shortage Rate TP (%)",          "Service"),
        ("UC_%",       "Capacity Utilization UC (%)",   "Capacity"),
        ("Stab",       "Stability",                     "Capacity"),
        ("TRA_%",      "Rework Rate TRA (%)",           "Quality"),
        ("P_Good",     "Good Production P_good",        "Quality"),
        ("C_Total",    "Total Cost C_total (€)",        "Costs"),
        ("CU",         "Unit Cost CU (€/unit)",         "Costs"),
        ("C_Penurie",  "Shortage Cost C_pénurie (€)",   "Costs"),
        ("C_Prod",     "Production Cost C_prod (€)",    "Costs"),
        ("E_Tot_kWh",  "Total Energy E_tot (kWh)",      "Energy"),
        ("IECP_eur",   "Energy-Carbon Cost IECP (€)",   "Energy"),
    ],
    "Service":  ["TS_%", "TP_%"],
    "Capacity": ["UC_%", "Stab"],
    "Quality":  ["TRA_%", "P_Good"],
    "Costs":    ["C_Total", "CU", "C_Penurie", "C_Prod"],
    "Energy":   ["E_Tot_kWh", "IECP_eur"],
}

_KPI_META = {entry[0]: (entry[1], entry[2]) for entry in KPI_CATEGORIES["All"]}


def render_kpi_dashboard():
    if not enter_page_once("kpi_dashboard"):
        return

    page_header(
        "KPIs Dashboard",
        "Visualize and compare all production KPIs across scenarios by category."
    )

    if "results_df" not in st.session_state:
        st.markdown(
            """
            <div class="clean-warning">
                Please run the optimization in Decision Support first to generate KPI results.
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to Decision Support", type="primary", key="kpi_go_decision_button"):
            go_to("Decision Support")
        return

    results_df = st.session_state.results_df

    # ── Category filter buttons ───────────────────────────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Filter by category")
    cat_cols = st.columns(6)
    categories = ["All", "Service", "Capacity", "Quality", "Costs", "Energy"]
    for i, cat in enumerate(categories):
        with cat_cols[i]:
            is_active = st.session_state.kpi_category == cat
            if st.button(
                cat,
                key=f"kpi_cat_{cat.lower()}_btn",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state.kpi_category = cat
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    selected_cat = st.session_state.kpi_category

    # ── Resolve which KPIs to show ────────────────────────────────────────────
    if selected_cat == "All":
        kpi_cols = [entry[0] for entry in KPI_CATEGORIES["All"]]
    else:
        kpi_cols = KPI_CATEGORIES[selected_cat]

    kpi_cols = [c for c in kpi_cols if c in results_df.columns]

    if not kpi_cols:
        st.info("No KPI data available for the selected category.")
        return

    best_idx = results_df["Score"].idxmax()
    best_scenario = int(results_df.loc[best_idx, "Scenario"])

    # ── KPI Bar charts ────────────────────────────────────────────────────────
    if plotly_available():
        n_kpis = len(kpi_cols)
        cols_per_row = 2
        rows_needed = (n_kpis + cols_per_row - 1) // cols_per_row

        for row_idx in range(rows_needed):
            chart_cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                kpi_idx = row_idx * cols_per_row + col_idx
                if kpi_idx >= n_kpis:
                    break
                kpi_col = kpi_cols[kpi_idx]
                label, _ = _KPI_META.get(kpi_col, (kpi_col, ""))
                with chart_cols[col_idx]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    colors = [
                        "#1d4ed8" if int(r["Scenario"]) == best_scenario else "#93c5fd"
                        for _, r in results_df.iterrows()
                    ]
                    fig = go.Figure(
                        go.Bar(
                            x=results_df["Scenario"].astype(str),
                            y=results_df[kpi_col],
                            marker_color=colors,
                            text=results_df[kpi_col].round(2),
                            textposition="outside",
                        )
                    )
                    fig.update_layout(
                        title=label,
                        xaxis_title="Scenario",
                        yaxis_title=label,
                        template="plotly_white",
                        margin=dict(l=10, r=10, t=45, b=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        # ── Radar chart: all scenarios (normalized) ───────────────────────────
        radar_kpis = ["TS_%", "UC_%", "Stab", "TRA_%"]
        radar_kpis = [c for c in radar_kpis if c in results_df.columns]

        if len(radar_kpis) >= 3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Scenario comparison — radar (normalized KPIs)")

            fig = go.Figure()
            theta_labels = [_KPI_META.get(c, (c, ""))[0] for c in radar_kpis]

            for _, srow in results_df.iterrows():
                # Normalize each radar KPI to [0, 1]
                values_radar = []
                for c in radar_kpis:
                    max_val = results_df[c].max()
                    v = float(srow[c]) / max_val if max_val > 0 else 0.0
                    values_radar.append(round(min(max(v, 0.0), 1.0), 4))

                fig.add_trace(
                    go.Scatterpolar(
                        r=values_radar + [values_radar[0]],
                        theta=theta_labels + [theta_labels[0]],
                        fill="toself",
                        opacity=0.75,
                        name=f"Scenario {int(srow['Scenario'])}",
                    )
                )

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_white",
                margin=dict(l=20, r=20, t=35, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Matplotlib fallback
        for kpi_col in kpi_cols:
            label, _ = _KPI_META.get(kpi_col, (kpi_col, ""))
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            colors_mpl = [
                "#1d4ed8" if int(r["Scenario"]) == best_scenario else "#93c5fd"
                for _, r in results_df.iterrows()
            ]
            ax.bar(results_df["Scenario"].astype(str), results_df[kpi_col], color=colors_mpl)
            ax.set_xlabel("Scenario")
            ax.set_ylabel(label)
            ax.set_title(label)
            style_plot(ax)
            close_plot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Full KPI table ────────────────────────────────────────────────────────
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Full KPI table")
    all_kpi_cols = [entry[0] for entry in KPI_CATEGORIES["All"]]
    table_cols = ["Scenario", "Score"] + [c for c in all_kpi_cols if c in results_df.columns]
    st.dataframe(results_df[table_cols], use_container_width=True)
    st.download_button(
        "Download KPI report",
        data=to_csv_bytes(results_df[table_cols]),
        file_name="kpi_dashboard_report.csv",
        mime="text/csv",
        type="secondary",
        key="kpi_download_button",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PAGE ROUTER
# =========================================================

if st.session_state.page == "Home":
    render_home()

elif st.session_state.page == "Upload Data":
    render_upload()

elif st.session_state.page == "Analysis":
    render_analysis()

elif st.session_state.page == "Prediction":
    render_prediction()

elif st.session_state.page == "Scenarios":
    render_scenarios()

elif st.session_state.page == "Decision Support":
    render_decision()

elif st.session_state.page == "KPIs Dashboard":
    render_kpi_dashboard()

