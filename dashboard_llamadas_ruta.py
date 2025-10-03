# streamlit run dashboard_llamadas_ruta.py
import os, io, math, base64
from datetime import date, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# CONFIG & THEME
# =========================
st.set_page_config(
    page_title="Dashboard de Llamadas de Ventas",
    layout="wide",
)

# =========================
# CARGA DE LOGOS
# =========================
def encode_image(path: str):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"❌ No se encontró la imagen en: {path}")
        return ""
    except Exception as e:
        st.error(f"❌ Error al cargar la imagen {path}: {e}")
        return ""

cun_path = "Data/cun.png"
cltiene_path = "Data/cltiene.png"
encoded_cun = encode_image(cun_path)
encoded_cltiene = encode_image(cltiene_path)

col1, col2 = st.columns(2)
with col1:
    if encoded_cun:
        st.markdown(
            f"""
            <style>.logo{{width:150px;height:auto;}}</style>
            <img class="logo" src="data:image/png;base64,{encoded_cun}" alt="Logo CUN">
            """,
            unsafe_allow_html=True
        )
with col2:
    if encoded_cltiene:
        st.markdown(
            f"""
            <style>.logo{{width:150px;height:auto;}}</style>
            <img class="logo" src="data:image/png;base64,{encoded_cltiene}" alt="Logo CLtiene">
            """,
            unsafe_allow_html=True
        )

CUSTOM_CSS = """
<style>
body, .stApp {
  background: radial-gradient(circle at 1px 1px, rgba(0,0,0,0.05) 1px, transparent 1px) 0 0 / 24px 24px;
}
.cun-header {
  background: linear-gradient(90deg, #007A33, #84BD00);
  color: white; padding: 18px 24px; border-radius: 14px;
  margin-bottom: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}
.cun-title { font-size: 28px; font-weight: 800; }
.cun-subtitle { opacity: 0.95; }
.kpi { background: white; border-radius: 14px; padding: 14px 16px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08); border: 1px solid rgba(0,0,0,0.05); }
.kpi h3 { font-size: 14px; margin: 0 0 6px 0; color: #666; font-weight: 600; }
.kpi .value { font-size: 26px; font-weight: 800; margin-top: 2px; }
.cun-card { background: white; border-radius: 14px; padding: 16px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08); border: 1px solid rgba(0,0,0,0.05); }
.dataframe tbody tr:hover { background: rgba(0,122,51,0.06); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

@st.cache_data(show_spinner=False)
def read_from_path(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
    return pd.read_excel(path)

@st.cache_data(show_spinner=False)
def read_from_bytes(content: bytes, ext: str) -> pd.DataFrame:
    bio = io.BytesIO(content)
    if ext == "csv":
        return pd.read_csv(bio, encoding="utf-8", encoding_errors="ignore")
    return pd.read_excel(bio)

def ensure_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    m = {}
    m["fecha"] = resolve_col(df, ["Fecha","fecha","FECHA","Fecha_Llamada","Fecha Llamada","CreatedAt","Date"])
    m["asesor"] = resolve_col(df, ["Asesor","Agente","Usuario","asesor","Agente_Nombre","Nombre Asesor"])
    m["tipo"] = resolve_col(df, ["Tipo","Canal","tipo"])
    m["clasificacion"] = resolve_col(df, ["Clasificación","Clasificacion","class","Categoria","Categoría"])
    m["puntaje"] = resolve_col(df, ["Puntaje","Puntaje Promedio","Score","Calificacion","Calificación"])
    m["confianza"] = resolve_col(df, ["Confianza","confidence","Probabilidad","Probabilidad_Pred"])
    m["subjetividad"] = resolve_col(df, ["Subjetividad","subjectivity","Subj"])
    m["neutralidad"] = resolve_col(df, ["Neutralidad","neutrality","Neutral"])
    m["polaridad"] = resolve_col(df, ["Polaridad","polarity","Sentiment","Sentimiento"])
    m["texto"] = resolve_col(df, ["Transcripcion","Transcripción","Texto","Mensaje","Contenido"])
    return m

def safe_mean(series: pd.Series) -> float:
    return float(series.dropna().mean()) if series is not None else float("nan")

# =========================
# LOAD DATA
# =========================
DEFAULT_PATH = "Data/CLtiene3-octubre-1"
df_raw = read_from_path(DEFAULT_PATH)

if df_raw.empty:
    st.error(f"No se encontró el archivo de datos en: {DEFAULT_PATH}. Verifica que exista en el repo con el mismo nombre y ruta.")
    st.stop()

df_raw = df_raw.copy()

# =========================
# SCHEMA & PREPROCESS
# =========================
schema = ensure_schema(df_raw)
df_raw.columns = df_raw.columns.str.strip().str.lower()

if schema.get("fecha"):
    col_fecha = schema["fecha"].lower()
    df_raw[col_fecha] = pd.to_datetime(df_raw[col_fecha], errors="coerce")
else:
    df_raw["__fecha__"] = pd.date_range("2025-01-01", periods=len(df_raw), freq="D")
    schema["fecha"] = "__fecha__"
    col_fecha = "__fecha__"

s_fecha = df_raw[col_fecha].dropna()
if not s_fecha.empty:
    min_d = s_fecha.min().date()
    max_d = s_fecha.max().date()
    if min_d == max_d:
        max_d = min_d + timedelta(days=1)
    default_range = (min_d, max_d)
else:
    hoy = date.today()
    default_range = (hoy - timedelta(days=7), hoy)

for k in ["puntaje","confianza","subjetividad","neutralidad","polaridad"]:
    if schema.get(k):
        df_raw[schema[k]] = pd.to_numeric(df_raw[schema[k]], errors="coerce")

# =========================
# SIDEBAR: FILTROS
# =========================
F = schema["fecha"]
A = schema["asesor"] or "__ASESOR__"
if A not in df_raw.columns:
    df_raw[A] = "Asesor Desconocido"
T = schema["tipo"]
C = schema["clasificacion"]
PUN = schema["puntaje"]; CON = schema["confianza"]; SUB = schema["subjetividad"]; NEU = schema["neutralidad"]; POL = schema["polaridad"]

with st.sidebar:
    st.markdown("### Filtros")

    date_range = st.date_input("Rango de fechas", value=default_range)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
    else:
        start = date_range
        end = date_range

    tipos = sorted([x for x in df_raw[T].dropna().unique()]) if T else []
    tipo_sel = st.multiselect("Tipo", tipos, default=tipos if tipos else [])

    asesores = sorted([x for x in df_raw[A].dropna().unique()])
    asesores_sel = st.multiselect("Asesor", asesores, default=asesores)

    clas = sorted([x for x in df_raw[C].dropna().unique()]) if C else []
    clas_sel = st.multiselect("Clasificación", clas, default=clas if clas else [])

mask = (pd.to_datetime(df_raw[F]).dt.date >= start) & (pd.to_datetime(df_raw[F]).dt.date <= end)
if T and len(tipo_sel) > 0: mask &= df_raw[T].isin(tipo_sel)
if C and len(clas_sel) > 0: mask &= df_raw[C].isin(clas_sel)
if len(asesores_sel) > 0: mask &= df_raw[A].isin(asesores_sel)
df = df_raw.loc[mask].copy()

# =========================
# HEADER
# =========================
st.markdown(
    f"""
    <div class="cun-header">
      <div class="cun-title">Dashboard de Llamadas de Ventas</div>
      <div class="cun-subtitle">Periodo: <b>{start.strftime('%Y-%m-%d')}</b> → <b>{end.strftime('%Y-%m-%d')}</b></div>
    </div>
    """, unsafe_allow_html=True
)

# =========================
# KPIs
# =========================
total_llamadas = len(df)
total_asesores = df[A].nunique()

kpi_cols = st.columns(6, gap="small")
def kpi(col, title, value, suffix=""):
    with col:
        st.markdown(f"""<div class="kpi"><h3>{title}</h3><div class="value">{value}{suffix}</div></div>""", unsafe_allow_html=True)

def safe_val(v, fmt):
    return fmt.format(v) if v == v and not math.isnan(v) else "—"

puntaje_prom = safe_mean(df[PUN]) if PUN and PUN in df.columns else float("nan")
conf_prom = safe_mean(df[CON])*100 if CON and CON in df.columns else float("nan")
subj_prom = safe_mean(df[SUB])*100 if SUB and SUB in df.columns else float("nan")
neut_prom = safe_mean(df[NEU])*100 if NEU and NEU in df.columns else float("nan")

kpi(kpi_cols[0], "Puntaje Promedio", safe_val(puntaje_prom, "{:,.1f}"))
kpi(kpi_cols[1], "Confianza", safe_val(conf_prom, "{:,.1f}"), "%")
kpi(kpi_cols[2], "Subjetividad", safe_val(subj_prom, "{:,.1f}"), "%")
kpi(kpi_cols[3], "Neutralidad", safe_val(neut_prom, "{:,.1f}"), "%")
kpi(kpi_cols[4], "Conteo Llamadas", f"{total_llamadas:,}")
kpi(kpi_cols[5], "Conteo Asesores", f"{total_asesores:,}")

st.markdown(" ")

# =========================
# (Gráficas y tablas siguen igual que antes…)
# =========================
# Aquí irían tus gráficas de evolución temporal, barras, bubble chart y tabla

st.markdown("---")
st.caption("CUN Analytics · Streamlit + Plotly · © 2025")

