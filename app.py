"""
====================================================================
HDB Resale Price Predictor — Streamlit Web Application
====================================================================

Required files in the SAME folder as this script:
    lr_log_model.pkl        - trained LinearRegression (log target)
    onehot_encoder.pkl       - fitted OneHotEncoder (town/flat_type/flat_model)
    feature_columns.pkl      - exact column order the model expects
    hdb_reference_data.csv   - 5,000-row stratified sample, used for the
                               market-context chart & recent comparables
                               (produced in Section 6 of the notebook)

Assets required alongside this script (in an assets/ subfolder):
    assets/valuai_icon.png             - house mark only, square, transparent
    assets/valuai_logo_transparent.png - icon + "VALU.AI" wordmark, transparent
====================================================================
"""

# ────────────────────────────────────────────────────────────────────
# Section 1 — Imports
# ────────────────────────────────────────────────────────────────────
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

CAT_COLS = ["town", "flat_type", "flat_model"]
CURRENT_YEAR = datetime.now().year  # used to back-calculate lease commencement year


# ────────────────────────────────────────────────────────────────────
# Section 2 — Page configuration
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Valu.ai",
    page_icon=Image.open("assets/valuai_icon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)


# ────────────────────────────────────────────────────────────────────
# Section 3 — Load model artefacts & reference data (cached)
# ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    model = joblib.load("lr_log_model.pkl")
    encoder = joblib.load("onehot_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, encoder, feature_columns


@st.cache_data(show_spinner=False)
def load_reference_data():
    return pd.read_csv("hdb_reference_data.csv")


try:
    model, encoder, feature_columns = load_model_artifacts()
    reference_df = load_reference_data()
except FileNotFoundError:
    st.error(
        "Model files are missing from the app folder. Please make sure "
        "**lr_log_model.pkl**, **onehot_encoder.pkl**, **feature_columns.pkl** "
        "and **hdb_reference_data.csv** sit in the same directory as app.py "
        "(see Section 6 of Ikhlas_Code.ipynb to generate them)."
    )
    st.stop()

# Dropdown options are read straight from the fitted encoder. This
# guarantees the app can never offer a category the model wasn't
# trained on, without needing to keep a second hardcoded list in sync.
TOWN_OPTIONS = sorted(encoder.categories_[0])
FLAT_TYPE_OPTIONS = sorted(encoder.categories_[1])
FLAT_MODEL_OPTIONS = sorted(encoder.categories_[2])

# Raw values in the dataset are upper-case (e.g. "ANG MO KIO", "4 ROOM").
# Title-case them for display, but keep flat_model as-is since several
# categories are acronyms/abbreviations (e.g. "DBSS") that .title() would
# otherwise mangle into "Dbss".
town_label_to_raw = {t.title(): t for t in TOWN_OPTIONS}
flat_type_label_to_raw = {t.title(): t for t in FLAT_TYPE_OPTIONS}
flat_model_label_to_raw = {t: t for t in FLAT_MODEL_OPTIONS}

FLOOR_AREA_MIN, FLOOR_AREA_MAX = 52, 157        # from EDA describe() in 2.2

# The model was trained on storey_lower (the start of a fixed 3-floor band,
# e.g. "13 TO 15" -> 13), NOT a raw floor number. STOREY_MIN/MAX below are
# the real range of actual floors a flat can be on; the app still asks the
# user for their real floor (better UX, matches how buyers think), and
# storey_to_bin() converts it to the exact bin the model expects, so
# accuracy is unaffected by asking for the real floor instead of the bin.
STOREY_MIN, STOREY_MAX = 1, 46                  # from EDA describe() in 2.2
STOREY_BIN_SIZE = 3                             # HDB storey_range bands are fixed 3-floor bins

# remaining_lease_months is the precise feature the model trained on.
# The app collects it as (years, extra months) for a more natural input,
# then combines the two before prediction.
LEASE_YEARS_MIN, LEASE_YEARS_MAX = 40, 97       # from remaining_lease_months describe()
LEASE_MONTHS_MIN, LEASE_MONTHS_MAX = 0, 11      # leftover months on top of the whole-year part
LEASE_TERM_MONTHS = 99 * 12                     # HDB flats carry a 99-year lease


# ────────────────────────────────────────────────────────────────────
# Section 4 — Theme: Bootstrap 5 + Montserrat, matched to the Figma design
# ────────────────────────────────────────────────────────────────────
def inject_theme() -> None:
    # IMPORTANT: every line below is flush against the left margin (no
    # leading indentation). Streamlit's markdown renderer can misinterpret
    # an INDENTED <style> block — it treats the indentation as a code block
    # rather than raw HTML, which causes chunks of the CSS to leak onto the
    # page as visible text instead of being applied as styling. Keeping this
    # string unindented avoids that.
    theme_css = """<link href="https://cdn.jsdelivr.net/npm/[email protected]/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
--bg-main:#07101f; --bg-sidebar:#040c18; --bg-card:#0c1b30;
--accent:#3b82f6; --accent-dark:#1d4ed8;
--text-primary:#dce8f8; --text-muted:#6b8aad;
--border-blue:rgba(59,130,246,0.16);
}
/* Setting font-family on html/body only reaches elements THROUGH
   inheritance, and inheritance loses to any rule -- important or not --
   that targets an element directly. Streamlit's own stylesheet sets
   font-family directly on headings, paragraphs, dropdown text, etc., so
   that's what was actually showing before, not our Montserrat import.
   A single `*` rule targets every element directly instead, so nothing
   is left depending on inheritance. */
* { font-family:'Montserrat',sans-serif !important; }
/* Exception: Streamlit's built-in icons (sidebar collapse arrow, dropdown
   chevrons, etc.) are ligature text in Google's "Material Symbols" font --
   e.g. the literal text "keyboard_arrow_down" renders as a ▾ glyph only
   under that font. Forcing Montserrat here would turn icons into visible
   words, so hand this one element type its font back. */
[data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded' !important; }
.stApp{ background:var(--bg-main); }
section[data-testid="stSidebar"]{
background:var(--bg-sidebar) !important;
border-right:1px solid var(--border-blue);
}
section[data-testid="stSidebar"] * { color: var(--text-primary); }
section[data-testid="stSidebar"] label p { color: var(--text-muted) !important; font-weight:600; font-size:0.72rem; text-transform:uppercase; letter-spacing:.08em; }
h1,h2,h3,h4,h5,h6,p,span,div,label { color:var(--text-primary); }
[data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }
.stButton>button{
background:var(--accent); color:#fff !important; border:none; border-radius:10px;
font-weight:700; padding:0.6rem 0; box-shadow:0 4px 20px rgba(59,130,246,0.45);
transition:all .15s;
}
.stButton>button:hover{ background:var(--accent-dark); color:#fff !important; }
.stButton>button p { color:#fff !important; }
div[data-baseweb="select"] > div, .stRadio { background: transparent; }
/* DevTools confirmed data-baseweb="slider" doesn't exist in this Streamlit
   version — every rule above that used to target it was a silent no-op.
   [data-testid="stSlider"] is the stable wrapper Streamlit still guarantees
   regardless of internal markup; role="slider" and the inline height-styled
   div are the real thumb/track confirmed from the pasted HTML. */
[data-testid="stSlider"] div[role="slider"]{ background-color:var(--accent) !important; border-color:var(--accent) !important; box-shadow:0 0 0 4px rgba(59,130,246,0.2) !important; }
[data-testid="stSlider"] div[data-testid="stTickBarMin"], [data-testid="stSlider"] div[data-testid="stTickBarMax"]{ color:var(--text-muted) !important; background:transparent !important; }
/* The track is a single div (not separate filled/unfilled divs), so we
   paint it one solid colour rather than guess at a gradient syntax we
   can't see — simpler and avoids further blind guessing. */
[data-testid="stSlider"] div[style*="height"]:not([role="slider"]):not([data-testid="stTickBarMin"]):not([data-testid="stTickBarMax"]){ background:var(--accent) !important; background-image:none !important; }
.mono{ font-family:'Montserrat',sans-serif; font-variant-numeric: tabular-nums; }
.hdb-card{
border-top:1px solid var(--border-blue); padding-top:1.25rem; margin-top:1.25rem;
}
.hdb-hero{
background:linear-gradient(145deg,#0a1628 0%,#0d1e36 40%,#0f2144 100%);
border:1px solid rgba(59,130,246,0.22); border-radius:16px;
padding:1.75rem; box-shadow:0 8px 32px rgba(0,0,0,0.35); margin-bottom:1rem;
}
.hdb-badge{
background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.25);
border-radius:99px; padding:5px 14px; font-size:0.72rem; font-weight:700;
color:#60a5fa !important; display:inline-flex; align-items:center; gap:6px;
}
.hdb-badge span { color:#60a5fa !important; }
.hdb-stat{
background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.15);
border-radius:8px; padding:6px 12px;
}
.hdb-muted{ color:var(--text-muted) !important; font-size:0.8rem; }
.hdb-step{
padding:14px 16px; height:100%;
}
</style>"""
    st.markdown(theme_css, unsafe_allow_html=True)


inject_theme()


# ────────────────────────────────────────────────────────────────────
# Section 5 — Session state defaults
# ────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "town_label": "Tampines",
    "flat_type_label": "4 Room",
    "floor_area": 95,
    "storey_actual": 8,                 # real floor number the user selects
    "flat_model_label": "Model A",
    "remaining_lease_years": 72,
    "remaining_lease_extra_months": 0,  # leftover months on top of the years
}

for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

st.session_state.setdefault("app_state", "empty")   # "empty" | "predicted" | "updated"
st.session_state.setdefault("price", None)
st.session_state.setdefault("prev_price", None)


# ────────────────────────────────────────────────────────────────────
# Section 6 — Prediction logic (mirrors the notebook's sanity-check cell)
# ────────────────────────────────────────────────────────────────────
def storey_to_bin(storey: int) -> tuple[int, str]:
    """Maps a real floor number to the fixed 3-floor storey_range bin used
    in training (e.g. floor 14 -> storey_lower=13, label '13 TO 15').
    This is an exact mapping, not an approximation: HDB itself only ever
    records storey as one of these bins, so any floor within a bin was
    indistinguishable to the model at training time regardless."""
    storey_lower = ((storey - 1) // STOREY_BIN_SIZE) * STOREY_BIN_SIZE + 1
    storey_upper = storey_lower + STOREY_BIN_SIZE - 1
    return storey_lower, f"{storey_lower:02d} TO {storey_upper:02d}"


def predict_price(town_raw, flat_type_raw, floor_area_sqm, storey_lower,
                   flat_model_raw, remaining_lease_months) -> float:
    """Builds a single-row input exactly like the training data, encodes
    it with the SAME fitted encoder used in the notebook, aligns columns
    to X_train's order, and returns the price in dollars (inverse of the
    log1p transform the model was trained on)."""

    # HDB flats carry a 99-year lease from lease_commence_date; back-solve
    # the commencement year from the precise remaining lease (in months).
    elapsed_months = LEASE_TERM_MONTHS - remaining_lease_months
    lease_commence_date = CURRENT_YEAR - round(elapsed_months / 12)

    raw_row = pd.DataFrame({
        "town": [town_raw],
        "flat_type": [flat_type_raw],
        "floor_area_sqm": [floor_area_sqm],
        "flat_model": [flat_model_raw],
        "lease_commence_date": [lease_commence_date],
        "remaining_lease_months": [remaining_lease_months],
        "storey_lower": [storey_lower],
    })

    encoded = encoder.transform(raw_row[CAT_COLS])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(CAT_COLS))

    model_input = pd.concat(
        [raw_row.drop(columns=CAT_COLS).reset_index(drop=True), encoded_df], axis=1
    ).reindex(columns=feature_columns, fill_value=0)

    pred_log = model.predict(model_input)[0]
    return float(np.expm1(pred_log))


def get_market_bands(flat_type_raw: str, town_raw: str, price: float):
    """Real percentile bands from the reference sample (not fabricated
    multipliers), scoped to town+flat_type when there's enough data,
    otherwise falling back to flat_type only."""
    subset = reference_df[reference_df["flat_type"] == flat_type_raw]
    town_subset = subset[subset["town"] == town_raw]
    if len(town_subset) >= 15:
        subset = town_subset
    if subset.empty:
        subset = reference_df

    q = subset["resale_price"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    bands = [
        {"label": "10th pct.", "value": float(q[0.10]), "is_estimate": False},
        {"label": "25th pct.", "value": float(q[0.25]), "is_estimate": False},
        {"label": "Median", "value": float(q[0.50]), "is_estimate": False},
        {"label": "Your est.", "value": float(price), "is_estimate": True},
        {"label": "75th pct.", "value": float(q[0.75]), "is_estimate": False},
        {"label": "90th pct.", "value": float(q[0.90]), "is_estimate": False},
    ]
    bands.sort(key=lambda b: b["value"])
    return bands, len(subset)


def get_comparables(flat_type_raw: str, town_raw: str, n: int = 3) -> pd.DataFrame:
    """Real recent transactions from the reference sample, same town and
    flat type where possible, most recent first."""
    subset = reference_df[
        (reference_df["flat_type"] == flat_type_raw) & (reference_df["town"] == town_raw)
    ]
    if len(subset) < n:
        subset = reference_df[reference_df["flat_type"] == flat_type_raw]
    return subset.sort_values("month", ascending=False).head(n)


# ────────────────────────────────────────────────────────────────────
# Section 7 — Sidebar (inputs)
# ────────────────────────────────────────────────────────────────────
def render_sidebar() -> bool:
    with st.sidebar:
        st.markdown(
            """
            <div class="hdb-muted" style="text-transform:uppercase;letter-spacing:.1em;font-size:0.62rem;">
                    HDB Resale Estimator &middot; SG
            </div>
            <p class="hdb-muted" style="font-size:0.78rem;line-height:1.6; margin-top:6px;">
                Enter flat details below to get an instant fair-value estimate
                based on recent HDB resale transactions.
            </p>
            <hr style="border-color:var(--border-blue); margin:0.75rem 0 1.1rem;">
            """,
            unsafe_allow_html=True,
        )

        st.selectbox("Town", sorted(town_label_to_raw.keys()), key="town_label")
        st.radio("Flat Type", sorted(flat_type_label_to_raw.keys()),
                  key="flat_type_label", horizontal=True)
        st.slider("Floor Area (sqm)", FLOOR_AREA_MIN, FLOOR_AREA_MAX, key="floor_area")

        st.slider("Which floor is the flat on?", STOREY_MIN, STOREY_MAX, key="storey_actual")
        _, storey_range_preview = storey_to_bin(st.session_state["storey_actual"])
        st.caption(f"HDB records this as storey range **{storey_range_preview}**.")

        st.selectbox("Flat Model", FLAT_MODEL_OPTIONS, key="flat_model_label")

        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
        lease_col_yrs, lease_col_mos = st.columns(2)
        with lease_col_yrs:
            st.slider("Remaining Lease (yrs)", LEASE_YEARS_MIN, LEASE_YEARS_MAX,
                       key="remaining_lease_years")
        with lease_col_mos:
            st.number_input("+ Months", LEASE_MONTHS_MIN, LEASE_MONTHS_MAX,
                             key="remaining_lease_extra_months")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("Predict Price", width="stretch")

        if st.session_state["app_state"] != "empty":
            st.button("Reset", width="stretch", on_click=_reset_form)

    return predict_clicked


def _reset_form() -> None:
    """Runs BEFORE the rerun triggered by the Reset button (via on_click),
    which is required — Streamlit blocks writes to a widget-bound
    session_state key once that widget has already rendered in the
    current script run."""
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["app_state"] = "empty"
    st.session_state["price"] = None
    st.session_state["prev_price"] = None


# ────────────────────────────────────────────────────────────────────
# Section 8 — Main panel: topbar, empty state, result panel
# ────────────────────────────────────────────────────────────────────
def render_topbar() -> None:
    state = st.session_state["app_state"]
    badge = ""
    if state == "predicted":
        badge = '<span class="hdb-badge">&#9679; Estimate ready</span>'
    elif state == "updated":
        badge = '<span class="hdb-badge">&#9679; Price updated</span>'

    col_icon, col_title, col_badge = st.columns(
        [0.06, 0.7, 0.24], vertical_alignment="center"
    )
    with col_icon:
        st.image("assets/valuai_icon.png", width=36)
    with col_title:
        st.markdown(
            """
            <div style="font-weight:700; font-size:1.15rem; color:#dce8f8; letter-spacing:.01em; line-height:1.2;">Valu.ai</div>
            <div class="hdb-muted" style="font-size:0.75rem;">HDB Resale Price Predictor &middot; Singapore</div>
            """,
            unsafe_allow_html=True,
        )
    with col_badge:
        if badge:
            st.markdown(
                f'<div style="text-align:right;">{badge}</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div style="border-bottom:1px solid var(--border-blue); margin-bottom:16px;"></div>',
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="text-center py-4">
            <h3 style="font-weight:700;">Is the asking price fair?</h3>
            <p class="hdb-muted mx-auto" style="max-width:420px; line-height:1.6;">
                Fill in the flat details on the left. We'll give you an instant
                fair-value estimate based on recent HDB resale transactions
                across Singapore &mdash; no sign-up required.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    steps = [
        ("01", "Choose location", "Town & flat type drive the biggest price variation."),
        ("02", "Set flat specs", "Area, storey, model and lease all factor in."),
        ("03", "Get estimate", "Instant valuation with real market context."),
    ]
    for i, (col, (n, title, sub)) in enumerate(zip((col1, col2, col3), steps)):
        with col:
            border_style = "border-left:1px solid var(--border-blue);" if i > 0 else ""
            st.markdown(
                f"""
                <div class="hdb-step" style="{border_style}">
                    <div class="mono" style="color:var(--accent); font-weight:700; font-size:0.75rem;">{n}</div>
                    <div style="font-weight:700; font-size:0.85rem; margin:4px 0;">{title}</div>
                    <div class="hdb-muted" style="font-size:0.75rem; line-height:1.5;">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_market_chart(bands, n_comparables: int) -> None:
    labels = [b["label"] for b in bands]
    values = [b["value"] for b in bands]
    colors = ["#3b82f6" if b["is_estimate"] else "rgba(59,130,246,0.30)" for b in bands]

    fig = go.Figure(
        go.Bar(
            x=labels, y=values, marker_color=colors,
            hovertemplate="%{x}<br>S$%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Montserrat", color="#6b8aad", size=11),
        margin=dict(l=10, r=10, t=10, b=10), height=220,
        yaxis=dict(tickprefix="S$", tickformat=",.0f", gridcolor="rgba(59,130,246,0.08)"),
        xaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(f"Based on {n_comparables} comparable resale transactions in the 2025\u201326 sample.")


def render_result_panel() -> None:
    price = st.session_state["price"]
    prev_price = st.session_state["prev_price"]
    is_updated = st.session_state["app_state"] == "updated"

    town_raw = town_label_to_raw[st.session_state["town_label"]]
    flat_type_raw = flat_type_label_to_raw[st.session_state["flat_type_label"]]

    _, storey_range_display = storey_to_bin(st.session_state["storey_actual"])
    lease_display = f"{st.session_state['remaining_lease_years']}y {st.session_state['remaining_lease_extra_months']}m"

    delta_html = ""
    if is_updated and prev_price is not None:
        diff = price - prev_price
        if diff > 0:
            delta_html = f'<span style="color:#4ade80; font-size:1rem; margin-left:10px;">&#9650; +S${diff:,.0f}</span>'
        elif diff < 0:
            delta_html = f'<span style="color:#f87171; font-size:1rem; margin-left:10px;">&#9660; S${diff:,.0f}</span>'

    st.markdown(
        f"""
        <div class="hdb-hero">
            <div class="hdb-muted" style="text-transform:uppercase; letter-spacing:.08em; font-size:0.7rem;">Estimated resale price</div>
            <div class="mono" style="font-size:2.3rem; font-weight:700; color:#dce8f8;">
                S$ {price:,.0f}{delta_html}
            </div>
            <div class="d-flex flex-wrap gap-2 mt-3 pt-3" style="border-top:1px solid var(--border-blue);">
                <div class="hdb-stat"><div class="hdb-muted" style="font-size:0.62rem; text-transform:uppercase;">Model</div>
                    <div class="mono" style="font-size:0.82rem; color:#93c5fd;">{st.session_state['flat_model_label']}</div></div>
                <div class="hdb-stat"><div class="hdb-muted" style="font-size:0.62rem; text-transform:uppercase;">Lease</div>
                    <div class="mono" style="font-size:0.82rem; color:#93c5fd;">{lease_display}</div></div>
                <div class="hdb-stat"><div class="hdb-muted" style="font-size:0.62rem; text-transform:uppercase;">Storey</div>
                    <div class="mono" style="font-size:0.82rem; color:#93c5fd;">#{st.session_state['storey_actual']} ({storey_range_display})</div></div>
                <div class="hdb-stat"><div class="hdb-muted" style="font-size:0.62rem; text-transform:uppercase;">Area</div>
                    <div class="mono" style="font-size:0.82rem; color:#93c5fd;">{st.session_state['floor_area']} sqm</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hdb-card">', unsafe_allow_html=True)
    st.markdown(
        "<strong>Market context</strong><br>"
        "<span class='hdb-muted'>Where your estimate sits among similar resale transactions</span>",
        unsafe_allow_html=True,
    )
    bands, n_comp = get_market_bands(flat_type_raw, town_raw, price)
    render_market_chart(bands, n_comp)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hdb-card">', unsafe_allow_html=True)
    st.markdown("<strong>Recent comparables</strong> &mdash; <span class='hdb-muted'>same flat type &amp; town</span>",
                unsafe_allow_html=True)
    comps = get_comparables(flat_type_raw, town_raw)
    for _, row in comps.iterrows():
        diff = row["resale_price"] - price
        if diff > 0:
            arrow, color, txt = "&#9650;", "#4ade80", f"+S${diff:,.0f}"
        elif diff < 0:
            arrow, color, txt = "&#9660;", "#f87171", f"S${diff:,.0f}"
        else:
            arrow, color, txt = "", "#6b8aad", "at est."
        st.markdown(
            f"""
            <div class="d-flex justify-content-between align-items-center py-2"
                 style="border-bottom:1px solid rgba(59,130,246,0.07);">
                <div>
                    <div style="font-weight:600; font-size:0.85rem;">Blk {row['block']} {row['street_name']}</div>
                    <div class="hdb-muted" style="font-size:0.7rem;">Storey {row['storey_range']} &middot; {row['month']}</div>
                </div>
                <div class="text-end">
                    <div class="mono" style="font-weight:700; font-size:0.85rem;">S$ {row['resale_price']:,.0f}</div>
                    <div class="mono" style="font-size:0.7rem; color:{color};">{arrow} {txt}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="hdb-card" style="display:flex; gap:12px;">
            <div>
                <strong style="color:#93c5fd;">&#10003; Estimate generated</strong>
                <p class="hdb-muted" style="margin-top:6px; margin-bottom:0; line-height:1.6;">
                    A {st.session_state['flat_type_label'].lower()} in
                    <strong style="color:#93c5fd;">{st.session_state['town_label']}</strong> on floor
                    {st.session_state['storey_actual']} with {lease_display} remaining on the lease
                    typically transacts around
                    <strong class="mono" style="color:#dce8f8;">S$ {price:,.0f}</strong>.
                    Always verify with a licensed property agent or HDB before transacting.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="text-center hdb-muted" style="font-size:0.7rem; padding:18px 0 6px;">
            Estimates based on HDB Resale Price Index data (data.gov.sg, 2025\u201326) &middot; For reference only
        </div>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────
# Section 9 — App entry point
# ────────────────────────────────────────────────────────────────────
def main() -> None:
    predict_clicked = render_sidebar()

    if predict_clicked:
        try:
            town_raw = town_label_to_raw[st.session_state["town_label"]]
            flat_type_raw = flat_type_label_to_raw[st.session_state["flat_type_label"]]
            flat_model_raw = flat_model_label_to_raw[st.session_state["flat_model_label"]]

            storey_lower, _ = storey_to_bin(st.session_state["storey_actual"])
            remaining_lease_months = (
                st.session_state["remaining_lease_years"] * 12
                + st.session_state["remaining_lease_extra_months"]
            )

            new_price = predict_price(
                town_raw, flat_type_raw,
                st.session_state["floor_area"], storey_lower,
                flat_model_raw, remaining_lease_months,
            )

            st.session_state["prev_price"] = st.session_state["price"]
            st.session_state["price"] = new_price
            st.session_state["app_state"] = (
                "updated" if st.session_state["app_state"] != "empty" else "predicted"
            )
        except Exception:
            st.error(
                "Something went wrong generating the estimate. Please check "
                "your inputs and try again."
            )

    render_topbar()

    if st.session_state["app_state"] == "empty":
        render_empty_state()
    else:
        render_result_panel()

    render_footer()


if __name__ == "__main__":
    main()