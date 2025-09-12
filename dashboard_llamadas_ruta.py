# streamlit run dashboard_llamadas_ruta.py
import os
import math
from datetime import date
from typing import Dict, List, Optional
import base64
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# carga de logos 
# =========================
# La ruta de tu imagen
# Function to encode the image to Base64
def encode_image(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error(f"❌ Error: No se encontró la imagen en: {path}")
        return ""
    except Exception as e:
        st.error(f"❌ Error al cargar la imagen {path}: {e}")
        return ""

# Paths for both images
cun_path = r"Data\cun.png"
cltiene_path = r"Data\cltiene.png"

# Encode both images
encoded_cun = encode_image(cun_path)
encoded_cltiene = encode_image(cltiene_path)

# Create two columns with equal width
col1, col2 = st.columns(2)

# Display the first image in the first column (col1)
with col1:
    if encoded_cun:
        st.markdown(
            f"""
            <style>
            .logo {{
                width: 150px;
                height: auto;
            }}
            </style>
            <img class="logo" src="data:image/png;base64,{encoded_cun}" alt="Logo Cun">
            """,
            unsafe_allow_html=True
        )

# Display the second image in the second column (col2)
with col2:
    if encoded_cltiene:
        st.markdown(
            f"""
            <style>
            .logo {{
                width: 150px;
                height: auto;
            }}
            </style>
            <img class="logo" src="data:image/png;base64,{encoded_cltiene}" alt="Logo Cltiene">
            """,
            unsafe_allow_html=True
        )
# =========================
# CONFIG & THEME
# =========================
st.set_page_config(
    page_title="Dashboard de Llamadas de Ventas",
    page_icon="📞",
    layout="wide",
)

CUSTOM_CSS = """
<style>
body, .stApp {
  background: radial-gradient(circle at 1px 1px, rgba(0,0,0,0.05) 1px, transparent 1px) 0 0 / 24px 24px;
}
.cun-header {
  background: linear-gradient(90deg, #007A33, #84BD00);
  color: white;
  padding: 18px 24px;
  border-radius: 14px;
  margin-bottom: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.15);
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
st.write(CUSTOM_CSS, unsafe_allow_html=True)

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
def load_data(default_path: Optional[str] = None) -> pd.DataFrame:
    """
    1) Si 'default_path' existe, lee CSV/XLSX de esa ruta.
    2) Si no existe, permite subir archivo desde la barra lateral.
    3) Si no hay nada, usa un demo mínimo.
    """
    # 1) Ruta por defecto
    if default_path and os.path.exists(default_path):
        if default_path.lower().endswith(".csv"):
            return pd.read_csv(default_path, encoding="utf-8", encoding_errors="ignore")
        else:
            # requiere 'openpyxl' instalado
            return pd.read_excel(default_path)

    # 2) Subida manual
    upl = st.sidebar.file_uploader("Sube (.xlsx/.csv)", type=["xlsx", "csv"])
    if upl is not None:
        if upl.name.lower().endswith(".xlsx"):
            return pd.read_excel(upl)
        else:
            return pd.read_csv(upl, encoding="utf-8", encoding_errors="ignore")

    # 3) Demo
    st.info("Cargando un demo mínimo hasta que especifiques una ruta o subas un archivo…")
    demo = pd.DataFrame({
        "Fecha": pd.date_range("2025-01-01", periods=40, freq="D"),
        "Asesor": np.random.choice(["Ana Ruiz","Carlos Pérez","Luisa Fernanda","Juan Díaz","María Gómez","Jorge Rojas"], 40),
        "Tipo": np.random.choice(["Entrante","Saliente"], 40),
        "Clasificación": np.random.choice(["Venta","No Venta","Seguimiento"], 40),
        "Puntaje": np.random.uniform(50, 100, 40).round(1),
        "Confianza": np.random.uniform(0.6, 0.98, 40).round(3),
        "Subjetividad": np.random.uniform(0.05, 0.6, 40).round(3),
        "Neutralidad": np.random.uniform(0.1, 0.9, 40).round(3),
        "Polaridad": np.random.uniform(-1, 1, 40).round(3),
    })
    return demo

def ensure_schema(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    mapping["fecha"] = resolve_col(df, ["Fecha","fecha","FECHA","Fecha_Llamada","Fecha Llamada","CreatedAt","Date"])
    mapping["asesor"] = resolve_col(df, ["Asesor","Agente","Usuario","asesor","Agente_Nombre","Nombre Asesor"])
    mapping["tipo"] = resolve_col(df, ["Tipo","Canal","tipo"])
    mapping["clasificacion"] = resolve_col(df, ["Clasificación","Clasificacion","class","Categoria","Categoría"])
    mapping["puntaje"] = resolve_col(df, ["Puntaje","Puntaje Promedio","Score","Calificacion","Calificación"])
    mapping["confianza"] = resolve_col(df, ["Confianza","confidence","Probabilidad","Probabilidad_Pred"])
    mapping["subjetividad"] = resolve_col(df, ["Subjetividad","subjectivity","Subj"])
    mapping["neutralidad"] = resolve_col(df, ["Neutralidad","neutrality","Neutral"])
    mapping["polaridad"] = resolve_col(df, ["Polaridad","polarity","Sentiment","Sentimiento"])
    mapping["texto"] = resolve_col(df, ["Transcripcion","Transcripción","Texto","Mensaje","Contenido"])
    return mapping

def safe_mean(series: pd.Series) -> float:
    return float(series.dropna().mean()) if series is not None else float("nan")

# =========================
# LOAD DATA (con ruta manual)
# =========================
DEFAULT_PATH = r"CLtiene.xlsx"  # <-- CAMBIA ESTA RUTA A LA TUYA

with st.sidebar:
    st.markdown("### Origen de datos")
    ruta_manual = st.text_input("Ruta local (opcional)", value=DEFAULT_PATH, help="Ej: C:\\carpeta\\archivo.xlsx")
df_raw = load_data(ruta_manual.strip() if ruta_manual else DEFAULT_PATH).copy()

# =========================
# SCHEMA & PREPROCESS
# =========================
schema = ensure_schema(df_raw)

# Parse fecha
if schema["fecha"]:
    df_raw[schema["fecha"]] = pd.to_datetime(df_raw[schema["fecha"]], errors="coerce")
else:
    # Si no hay fecha, fabricar por índice
    df_raw["__Fecha__"] = pd.date_range("2025-01-01", periods=len(df_raw), freq="D")
    schema["fecha"] = "__Fecha__"

# A numérico
for k in ["puntaje","confianza","subjetividad","neutralidad","polaridad"]:
    if schema.get(k):
        df_raw[schema[k]] = pd.to_numeric(df_raw[schema[k]], errors="coerce")

# Intento de sentimiento si falta (opcional)
try:
    from textblob import TextBlob  # type: ignore
    if (not schema["polaridad"] or df_raw[schema["polaridad"]].isna().all()) and schema["texto"]:
        pol = df_raw[schema["texto"]].astype(str).apply(lambda t: TextBlob(t).sentiment.polarity if t else np.nan)
        colname = schema["polaridad"] or "Polaridad"
        df_raw[colname] = pol.astype(float)
        schema["polaridad"] = colname
    if (not schema["subjetividad"] or df_raw[schema["subjetividad"]].isna().all()) and schema["texto"]:
        sub = df_raw[schema["texto"]].astype(str).apply(lambda t: TextBlob(t).sentiment.subjectivity if t else np.nan)
        colname = schema["subjetividad"] or "Subjetividad"
        df_raw[colname] = sub.astype(float)
        schema["subjetividad"] = colname
except Exception:
    pass

# Fallbacks
if not schema["confianza"]:
    if schema["polaridad"]:
        df_raw["Confianza_sint"] = (1 - np.clip(np.abs(df_raw[schema["polaridad"]]), 0, 1)) * 0.6 + 0.3
        schema["confianza"] = "Confianza_sint"
    elif schema["puntaje"]:
        maxv = df_raw[schema["puntaje"]].max()
        if pd.notna(maxv) and maxv > 0:
            df_raw["Confianza_sint"] = np.clip((df_raw[schema["puntaje"]] / maxv), 0, 1) * 0.5 + 0.3
            schema["confianza"] = "Confianza_sint"

if not schema["neutralidad"] and schema["polaridad"]:
    df_raw["Neutralidad_sint"] = 1 - np.clip(np.abs(df_raw[schema["polaridad"]]), 0, 1)
    schema["neutralidad"] = "Neutralidad_sint"

# Nombres canónicos de conveniencia
F = schema["fecha"]
A = schema["asesor"] or "__ASESOR__"
if A not in df_raw.columns:
    df_raw[A] = "Asesor Desconocido"
T = schema["tipo"]
C = schema["clasificacion"]

PUN = schema["puntaje"]
CON = schema["confianza"]
SUB = schema["subjetividad"]
NEU = schema["neutralidad"]
POL = schema["polaridad"]

# =========================
# SIDEBAR FILTERS
# =========================
with st.sidebar:
    st.markdown("### Filtros")
    min_d = pd.to_datetime(df_raw[F]).min()
    max_d = pd.to_datetime(df_raw[F]).max()
    # Asegurar rango válido
    if isinstance(min_d, pd.Timestamp) and isinstance(max_d, pd.Timestamp) and not pd.isna(min_d) and not pd.isna(max_d):
        default_range = (min_d.date(), max_d.date())
    else:
        hoy = date.today()
        default_range = (hoy, hoy)
    start, end = st.date_input("Rango de fechas", value=default_range)

    tipos = sorted([x for x in df_raw[T].dropna().unique()]) if T else []
    tipo_sel = st.multiselect("Tipo", tipos, default=tipos if tipos else [])
    asesores = sorted([x for x in df_raw[A].dropna().unique()])
    asesores_sel = st.multiselect("Asesor", asesores, default=asesores)
    clas = sorted([x for x in df_raw[C].dropna().unique()]) if C else []
    clas_sel = st.multiselect("Clasificación", clas, default=clas if clas else [])

mask = (pd.to_datetime(df_raw[F]).dt.date >= start) & (pd.to_datetime(df_raw[F]).dt.date <= end)
if T and len(tipo_sel) > 0: mask &= df_raw[T].isin(tipo_sel)
if C and len(clas_sel) > 0: mask &= df_raw[C].isin(clas_sel)
if len(asesores_sel) > 0:   mask &= df_raw[A].isin(asesores_sel)
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

puntaje_prom = safe_mean(df[PUN]) if PUN and PUN in df.columns else float("nan")
conf_prom = safe_mean(df[CON])*100 if CON and CON in df.columns else float("nan")
subj_prom = safe_mean(df[SUB])*100 if SUB and SUB in df.columns else float("nan")
neut_prom = safe_mean(df[NEU])*100 if NEU and NEU in df.columns else float("nan")

kpi(kpi_cols[0], "Puntaje Promedio", f"{puntaje_prom:,.1f}" if not math.isnan(puntaje_prom) else "—")
kpi(kpi_cols[1], "Confianza", f"{conf_prom:,.1f}" if not math.isnan(conf_prom) else "—", "%")
kpi(kpi_cols[2], "Subjetividad", f"{subj_prom:,.1f}" if not math.isnan(subj_prom) else "—", "%")
kpi(kpi_cols[3], "Neutralidad", f"{neut_prom:,.1f}" if not math.isnan(neut_prom) else "—", "%")
kpi(kpi_cols[4], "Conteo Llamadas", f"{total_llamadas:,}")
kpi(kpi_cols[5], "Conteo Asesores", f"{total_asesores:,}")

st.markdown(" ")

# =========================
# ROW 1: Evolución temporal + Barras por asesor
# =========================
c1, c2 = st.columns((1.15, 1), gap="large")

with c1:
    st.markdown("#### Evolución de Indicadores Emocionales en el Tiempo")
    df_ts = df.copy()
    df_ts["__Fecha__"] = pd.to_datetime(df_ts[F]).dt.date
    agg_cols = {}
    label_map = {}
    for (col, label) in [(CON,"Confianza"), (SUB,"Subjetividad"), (NEU,"Neutralidad")]:
        if col and col in df_ts.columns:
            agg_cols[col] = "mean"
            label_map[col] = label

    if len(agg_cols) == 0:
        st.info("No hay columnas de Confianza/Subjetividad/Neutralidad disponibles para graficar.")
    else:
        ts = df_ts.groupby("__Fecha__").agg(agg_cols).reset_index().rename(columns=label_map)
        ts_long = ts.melt(id_vars="__Fecha__", var_name="Indicador", value_name="Valor")
        fig = px.line(ts_long, x="__Fecha__", y="Valor", color="Indicador", markers=True)
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("#### Desempeño y Polaridad Promedio por Asesor")
    agg_cols = {}
    if PUN and PUN in df.columns: agg_cols[PUN] = "mean"
    if POL and POL in df.columns: agg_cols[POL] = "mean"
    if len(agg_cols) == 0:
        st.info("No hay columnas de Puntaje/Polaridad disponibles para graficar.")
    else:
        per_asesor = df.groupby(A).agg(agg_cols).reset_index()
        per_asesor = per_asesor.sort_values(by=list(agg_cols.keys())[0], ascending=False).head(8)
        rename_map = {}
        if PUN in per_asesor.columns: rename_map[PUN] = "Puntaje"
        if POL in per_asesor.columns: rename_map[POL] = "Polaridad"
        per_asesor = per_asesor.rename(columns=rename_map)
        long_bars = per_asesor.melt(id_vars=A, var_name="Métrica", value_name="Valor")
        fig2 = px.bar(long_bars, x=A, y="Valor", color="Métrica", barmode="group")
        fig2.update_layout(height=300, margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Asesor")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# ROW 2: Bubble chart + Tabla
# =========================
c3, c4 = st.columns((1, 1.1), gap="large")

with c3:
    st.markdown("#### Relación de Subjetividad y Confianza por Asesor")
    if not all([col and col in df.columns for col in [SUB, CON]]):
        st.info("No hay columnas suficientes para este gráfico (se requieren Subjetividad y Confianza).")
    else:
        bub = (df.groupby(A)
                 .agg({SUB:"mean", CON:"mean", F:"count"})
                 .reset_index()
                 .rename(columns={SUB:"Subjetividad", CON:"Confianza", F:"Llamadas"}))
        fig3 = px.scatter(
            bub, x="Subjetividad", y="Confianza", size="Llamadas", color=A,
            hover_name=A, size_max=40
        )
        fig3.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10), xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.markdown("#### Indicadores Clave por Asesor")
    agg = {}
    if CON in df.columns: agg[CON] = "mean"
    if SUB in df.columns: agg[SUB] = "mean"
    if NEU in df.columns: agg[NEU] = "mean"
    if POL in df.columns: agg[POL] = "mean"
    agg[F] = "count"

    tabla = (df.groupby(A).agg(agg).reset_index())
    rename = {}
    if CON in tabla.columns: rename[CON] = "Confianza"
    if SUB in tabla.columns: rename[SUB] = "Subjetividad"
    if NEU in tabla.columns: rename[NEU] = "Neutralidad"
    if POL in tabla.columns: rename[POL] = "Polaridad"
    rename[F] = "Llamadas"
    tabla = tabla.rename(columns=rename)

    for pcol in ["Confianza","Subjetividad","Neutralidad"]:
        if pcol in tabla.columns:
            tabla[pcol] = (tabla[pcol]*100).round(2)
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
        use_container_width=True
    )
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="llamadas_filtradas.csv",
        mime="text/csv",
        type="secondary",
        use_container_width=True
    )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("CUN Analytics · Streamlit + Plotly · © 2025")
