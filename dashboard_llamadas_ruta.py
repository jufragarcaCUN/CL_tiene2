# streamlit run dashboard_llamadas_ruta.py
import os, io, math, base64
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Dashboard de Llamadas de Ventas", layout="wide")

def encode_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
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
            unsafe_allow_html=True,
        )
with col2:
    if encoded_cltiene:
        st.markdown(
            f"""
            <style>.logo{{width:150px;height:auto;}}</style>
            <img class="logo" src="data:image/png;base64,{encoded_cltiene}" alt="Logo CLtiene">
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

def resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        lc = cand.lower()
        if lc in cols:
            return cols[lc]
    return None

@st.cache_data(show_spinner=False)
def read_from_path(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
    return pd.read_excel(path)

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

def to_percent(x: pd.Series) -> pd.Series:
    return (x * 100).round(2)

DEFAULT_PATH = "Data/CLtiene3-octubre-1"
df_raw = read_from_path(DEFAULT_PATH)
if df_raw.empty:
    st.error(f"No se encontró el archivo de datos en: {DEFAULT_PATH}.")
    st.stop()
df_raw = df_raw.copy()

schema = ensure_schema(df_raw)
df_raw.columns = df_raw.columns.str.strip().str.lower()
schema = {k: (v.lower() if isinstance(v, str) else v) for k, v in schema.items()}

if schema.get("fecha"):
    col_fecha = schema["fecha"]
    df_raw[col_fecha] = pd.to_datetime(df_raw[col_fecha], errors="coerce")
else:
    col_fecha = "__fecha__"
    df_raw[col_fecha] = pd.date_range("2025-01-01", periods=len(df_raw), freq="D")
    schema["fecha"] = col_fecha

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
    colk = schema.get(k)
    if colk and colk in df_raw.columns:
        df_raw[colk] = pd.to_numeric(df_raw[colk], errors="coerce")

F = schema["fecha"]
A = schema["asesor"] or "__asesor__"
if A not in df_raw.columns:
    df_raw[A] = "Asesor Desconocido"
T = schema.get("tipo")
C = schema.get("clasificacion")
PUN = schema.get("puntaje")
CON = schema.get("confianza")
SUB = schema.get("subjetividad")
NEU = schema.get("neutralidad")
POL = schema.get("polaridad")

with st.sidebar:
    st.markdown("### Filtros")
    date_sel = st.date_input(
        "Rango de fechas",
        value=(default_range[0], default_range[1]),
        min_value=default_range[0],
        max_value=default_range[1],
        key="rango_fechas_v4"
    )
    if isinstance(date_sel, (tuple, list)) and len(date_sel) == 2:
        start, end = date_sel[0], date_sel[1]
    else:
        start = end = date_sel
    if hasattr(start, "date"):
        try:
            start = start.date()
        except:
            pass
    if hasattr(end, "date"):
        try:
            end = end.date()
        except:
            pass

    tipos = sorted([x for x in df_raw[T].dropna().unique()]) if T and T in df_raw.columns else []
    tipo_sel = st.multiselect("Tipo", tipos, default=tipos if tipos else [])

    asesores = sorted([x for x in df_raw[A].dropna().unique()])
    asesores_sel = st.multiselect("Asesor", asesores, default=asesores)

    clas = sorted([x for x in df_raw[C].dropna().unique()]) if C and C in df_raw.columns else []
    clas_sel = st.multiselect("Clasificación", clas, default=clas if clas else [])

fecha_series = pd.to_datetime(df_raw[F], errors="coerce")
mask = (fecha_series.dt.date >= start) & (fecha_series.dt.date <= end)
if T and T in df_raw.columns and len(tipo_sel) > 0:
    mask &= df_raw[T].isin(tipo_sel)
if C and C in df_raw.columns and len(clas_sel) > 0:
    mask &= df_raw[C].isin(clas_sel)
if len(asesores_sel) > 0:
    mask &= df_raw[A].isin(asesores_sel)
df = df_raw.loc[mask].copy()

st.markdown(
    f"""
    <div class="cun-header">
      <div class="cun-title">Dashboard de Llamadas de Ventas</div>
      <div class="cun-subtitle">Periodo: <b>{start.strftime('%Y-%m-%d')}</b> → <b>{end.strftime('%Y-%m-%d')}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

total_llamadas = len(df)
total_asesores = df[A].nunique()

kpi_cols = st.columns(6, gap="small")
def kpi(col, title, value, suffix=""):
    with col:
        st.markdown(f"""<div class="kpi"><h3>{title}</h3><div class="value">{value}{suffix}</div></div>""", unsafe_allow_html=True)

def safe_val(v, fmt):
    return fmt.format(v) if v == v and not math.isnan(v) else "—"

puntaje_prom = safe_mean(df[PUN]) if PUN and PUN in df.columns else float("nan")
conf_prom = safe_mean(df[CON]) * 100 if CON and CON in df.columns else float("nan")
subj_prom = safe_mean(df[SUB]) * 100 if SUB and SUB in df.columns else float("nan")
neut_prom = safe_mean(df[NEU]) * 100 if NEU and NEU in df.columns else float("nan")

kpi(kpi_cols[0], "Puntaje Promedio", safe_val(puntaje_prom, "{:,.1f}"))
kpi(kpi_cols[1], "Confianza", safe_val(conf_prom, "{:,.1f}"), "%")
kpi(kpi_cols[2], "Subjetividad", safe_val(subj_prom, "{:,.1f}"), "%")
kpi(kpi_cols[3], "Neutralidad", safe_val(neut_prom, "{:,.1f}"), "%")
kpi(kpi_cols[4], "Conteo Llamadas", f"{total_llamadas:,}")
kpi(kpi_cols[5], "Conteo Asesores", f"{total_asesores:,}")

st.markdown(" ")

c1, c2 = st.columns((1.15, 1), gap="large")

with c1:
    st.markdown("#### Evolución de Indicadores Emocionales en el Tiempo")
    df_ts = df.copy()
    df_ts["__fecha__plot"] = pd.to_datetime(df_ts[F], errors="coerce").dt.date
    agg_cols, label_map = {}, {}
    for (col, label) in [(CON, "Confianza"), (SUB, "Subjetividad"), (NEU, "Neutralidad")]:
        if col and col in df_ts.columns:
            agg_cols[col] = "mean"
            label_map[col] = label
    if len(agg_cols) == 0:
        st.info("No hay columnas de Confianza/Subjetividad/Neutralidad disponibles para graficar.")
    else:
        ts = df_ts.groupby("__fecha__plot").agg(agg_cols).reset_index().rename(columns=label_map)
        ts_long = ts.melt(id_vars="__fecha__plot", var_name="Indicador", value_name="Valor")
        fig = px.line(ts_long, x="__fecha__plot", y="Valor", color="Indicador", markers=True)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("#### Desempeño y Polaridad Promedio por Asesor")
    agg_cols = {}
    if PUN and PUN in df.columns:
        agg_cols[PUN] = "mean"
    if POL and POL in df.columns:
        agg_cols[POL] = "mean"
    if len(agg_cols) == 0:
        st.info("No hay columnas de Puntaje/Polaridad disponibles para graficar.")
    else:
        per_asesor = df.groupby(A).agg(agg_cols).reset_index()
        by_field = list(agg_cols.keys())[0]
        per_asesor = per_asesor.sort_values(by=by_field, ascending=False).head(8)
        rename_map = {}
        if PUN in per_asesor.columns:
            rename_map[PUN] = "Puntaje"
        if POL in per_asesor.columns:
            rename_map[POL] = "Polaridad"
        per_asesor = per_asesor.rename(columns=rename_map)
        long_bars = per_asesor.melt(id_vars=A, var_name="Métrica", value_name="Valor")
        fig2 = px.bar(long_bars, x=A, y="Valor", color="Métrica", barmode="group")
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Asesor")
        st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns((1, 1.1), gap="large")

with c3:
    st.markdown("#### Relación de Subjetividad y Confianza por Asesor")
    if not all([col and col in df.columns for col in [SUB, CON]]):
        st.info("No hay columnas suficientes para este gráfico (se requieren Subjetividad y Confianza).")
    else:
        bub = (
            df.groupby(A)
            .agg({SUB: "mean", CON: "mean", F: "count"})
            .reset_index()
            .rename(columns={SUB: "Subjetividad", CON: "Confianza", F: "Llamadas"})
        )
        fig3 = px.scatter(bub, x="Subjetividad", y="Confianza", size="Llamadas", color=A, hover_name=A, size_max=40)
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown("#### Indicadores Clave por Asesor")
    agg = {}
    if CON and CON in df.columns:
        agg[CON] = "mean"
    if SUB and SUB in df.columns:
        agg[SUB] = "mean"
    if NEU and NEU in df.columns:
        agg[NEU] = "mean"
    if POL and POL in df.columns:
        agg[POL] = "mean"
    agg[F] = "count"

    tabla = df.groupby(A).agg(agg).reset_index()
    rename = {}
    if CON and CON in tabla.columns:
        rename[CON] = "Confianza"
    if SUB and SUB in tabla.columns:
        rename[SUB] = "Subjetividad"
    if NEU and NEU in tabla.columns:
        rename[NEU] = "Neutralidad"
    if POL and POL in tabla.columns:
        rename[POL] = "Polaridad"
    rename[F] = "Llamadas"
    tabla = tabla.rename(columns=rename)

    for pcol in ["Confianza", "Subjetividad", "Neutralidad"]:
        if pcol in tabla.columns:
            tabla[pcol] = to_percent(tabla[pcol])
    if "Polaridad" in tabla.columns:
        tabla["Polaridad"] = tabla["Polaridad"].round(3)
    if "Llamadas" in tabla.columns:
        tabla["Llamadas"] = tabla["Llamadas"].astype(int)

    st.dataframe(tabla, use_container_width=True, height=320)
    st.download_button(
        "⬇️ Descargar tabla por asesor (CSV)",
        data=tabla.to_csv(index=False).encode("utf-8"),
        file_name="indicadores_por_asesor.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="llamadas_filtradas.csv",
        mime="text/csv",
        type="secondary",
        use_container_width=True,
    )

st.markdown("---")
st.caption("CUN Analytics · Streamlit + Plotly · © 2025")
