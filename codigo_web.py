import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(page_title="Dashboard CPK TDR", layout="wide")

# --- Cargar archivo CSV ---
st.sidebar.title("📁 Cargar archivo")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV con datos de TDR", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    columnas_necesarias = ["Unidad", "Flota", "Semana_datetime", "CPK total", "kmstotales"]
    if not all(col in df.columns for col in columnas_necesarias):
        st.error("⚠️ El archivo no contiene todas las columnas necesarias.")
        st.stop()

    df["Unidad"] = df["Unidad"].astype(str)
    df["Flota"] = df["Flota"].astype(str)
    df["Semana_str"] = pd.to_datetime(df["Semana_datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["CPK total"] = pd.to_numeric(df["CPK total"], errors="coerce")
    df = df[df["CPK total"].notna() & df["CPK total"].apply(np.isfinite)]

    # --- Filtrar solo las primeras 10 Unidades y 10 Flotas ---
    top_unidades = df["Unidad"].dropna().unique()[:10]
    top_flotas = df["Flota"].dropna().unique()[:10]
    df = df[df["Unidad"].isin(top_unidades) & df["Flota"].isin(top_flotas)]

else:
    st.warning("Sube un archivo para visualizar el dashboard.")
    st.stop()

# --- Filtros estilo dropdown limpio ---
st.sidebar.title("🧰 Filtros")

opciones_unidades = sorted(df["Unidad"].dropna().unique())
opciones_flotas = sorted(df["Flota"].dropna().unique())

with st.sidebar.expander("🔧 Filtro por Unidades", expanded=False):
    unidades_sel = st.multiselect("Selecciona una o más unidades:", opciones_unidades, default=opciones_unidades)

with st.sidebar.expander("🚚 Filtro por Flotas", expanded=False):
    flotas_sel = st.multiselect("Selecciona una o más flotas:", opciones_flotas, default=opciones_flotas)

cpk_min, cpk_max = float(df["CPK total"].min()), float(df["CPK total"].max())
rango_cpk = st.sidebar.slider("Rango de CPK:", cpk_min, cpk_max, (cpk_min, cpk_max))

# --- Aplicar filtros ---
df_filtrado = df[
    (df["Unidad"].isin(unidades_sel)) &
    (df["Flota"].isin(flotas_sel)) &
    (df["CPK total"].between(*rango_cpk))
]

# --- Navegación de vistas ---
st.sidebar.title("📊 Visualizaciones")
vista = st.sidebar.radio("Selecciona una vista:", [
    "Boxplot CPK por Unidad",
    "Barplot CPK por Unidad",
    "Heatmap CPK por Semana",
    "Tendencia CPK en el Tiempo",
    "Boxplot por Flota",
    "Barplot por Flota",
    "Violin Plot por Flota",
    "Scatter CPK vs Km"
])

# --- Visualizaciones ---
if vista == "Boxplot CPK por Unidad":
    st.title("Boxplot de CPK total por Unidad")
    fig = px.box(df_filtrado, x="Unidad", y="CPK total", points="outliers")
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)

elif vista == "Barplot CPK por Unidad":
    st.title("Promedio de CPK total por Unidad")
    df_bar = df_filtrado.groupby("Unidad")["CPK total"].mean().reset_index().sort_values("CPK total", ascending=False)
    fig = px.bar(df_bar, x="Unidad", y="CPK total", color="CPK total")
    st.plotly_chart(fig, use_container_width=True)

elif vista == "Heatmap CPK por Semana":
    st.title("Heatmap de CPK total por Unidad y Semana")
    df_pivot = df_filtrado.pivot_table(index="Unidad", columns="Semana_str", values="CPK total", aggfunc="mean")
    fig = px.imshow(df_pivot, aspect="auto", color_continuous_scale="Viridis",
                    labels=dict(x="Semana", y="Unidad", color="CPK total"))
    st.plotly_chart(fig, use_container_width=True)

elif vista == "Tendencia CPK en el Tiempo":
    st.title("Tendencia semanal de CPK total")
    df_trend = df_filtrado.groupby("Semana_str")["CPK total"].mean().reset_index()
    fig = px.line(df_trend, x="Semana_str", y="CPK total", markers=True)

    x = np.arange(len(df_trend))
    y = df_trend["CPK total"].values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    trend_line = model.predict(X)

    fig.add_trace(go.Scatter(x=df_trend["Semana_str"], y=trend_line,
                             mode="lines", name="Tendencia", line=dict(color="red")))
    st.plotly_chart(fig, use_container_width=True)

elif vista == "Boxplot por Flota":
    st.title("Boxplot de CPK total por Flota")
    fig = px.box(df_filtrado, x="Flota", y="CPK total", points="outliers", color="Flota")
    st.plotly_chart(fig, use_container_width=True)

elif vista == "Barplot por Flota":
    st.title("Promedio de CPK total por Flota")
    df_flota = df_filtrado.groupby("Flota")["CPK total"].mean().reset_index().sort_values("CPK total", ascending=False)
    fig = px.bar(df_flota, x="Flota", y="CPK total", color="CPK total")
    st.plotly_chart(fig, use_container_width=True)

elif vista == "Violin Plot por Flota":
    st.title("Distribución de CPK total por Flota")
    fig = px.violin(df_filtrado, x="Flota", y="CPK total", box=True, points="all", color="Flota")
    st.plotly_chart(fig, use_container_width=True)

elif vista == "Scatter CPK vs Km":
    st.title("Relación entre CPK total y Km recorridos")
    df_scatter = df_filtrado.groupby(["Unidad", "Flota"]).agg({
        "CPK total": "mean", "kmstotales": "sum"
    }).reset_index()
    fig = px.scatter(df_scatter, x="CPK total", y="kmstotales", color="Flota", size="kmstotales",
                     hover_data=["Unidad"])
    st.plotly_chart(fig, use_container_width=True)
