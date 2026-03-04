"""
Comparador de Espectros de Respuesta — NSR-10
==============================================
Aplicación Streamlit para calcular el espectro de diseño según
el Título A, Capítulo A.2.6 de la NSR-10 y compararlo con un
espectro proporcionado por el usuario.

Ejecutar:
    streamlit run espectro_nsr10.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURACIÓN DE PÁGINA                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="Espectro de Diseño NSR-10",
    page_icon="🏗️",
    layout="wide",
)

st.title(" Comparador de Espectros de Respuesta — NSR-10")
st.markdown(
    "Cálculo del **espectro de diseño** según el **Título A, Capítulo A.2.6 "
    "de la NSR-10** y comparación con un espectro dado."
)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  TABLAS NORMATIVAS  (NSR-10 A.2.4)                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

AA_BP = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
AV_BP = np.array([0.05, 0.10, 0.15, 0.20, 0.25])

# Tabla A.2.4-3 — Coeficiente de sitio Fa
FA_TABLE = {
    "A": [0.80, 0.80, 0.80, 0.80, 0.80],
    "B": [1.00, 1.00, 1.00, 1.00, 1.00],
    "C": [1.20, 1.20, 1.10, 1.00, 1.00],
    "D": [1.60, 1.40, 1.20, 1.10, 1.00],
    "E": [2.50, 1.70, 1.20, 0.90, 0.90],
}

# Tabla A.2.4-4 — Coeficiente de sitio Fv
FV_TABLE = {
    "A": [0.80, 0.80, 0.80, 0.80, 0.80],
    "B": [1.00, 1.00, 1.00, 1.00, 1.00],
    "C": [1.70, 1.60, 1.50, 1.40, 1.30],
    "D": [2.40, 2.00, 1.80, 1.60, 1.50],
    "E": [3.50, 3.40, 2.80, 2.40, 2.40],
}

# Tabla A.2.5-1 — Coeficiente de importancia I
I_FACTORS = {
    "I  – Ocupación normal": 1.00,
    "II – Ocupación especial": 1.10,
    "III – Atención a la comunidad": 1.25,
    "IV – Edificaciones indispensables": 1.50,
}

SOIL_DESC = {
    "A": "Roca competente  (Vs ≥ 1500 m/s)",
    "B": "Roca de rigidez media  (760 < Vs ≤ 1500 m/s)",
    "C": "Suelos muy densos / roca blanda  (360 < Vs ≤ 760 m/s)",
    "D": "Suelos rígidos  (180 < Vs ≤ 360 m/s)",
    "E": "Suelos blandos  (Vs < 180 m/s)",
}

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FUNCIONES DE CÁLCULO                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def interp_coeff(value: float, breakpoints: np.ndarray, table_row: list) -> float:
    """Interpolación lineal según tablas A.2.4-3 / A.2.4-4."""
    return float(np.interp(value, breakpoints, table_row))


def compute_spectrum(
    aa: float,
    av: float,
    fa: float,
    fv: float,
    importance: float,
    T: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """
    Espectro de diseño NSR-10 — A.2.6.

    Parámetros
    ----------
    aa, av : Coeficientes de aceleración y velocidad pico efectiva.
    fa, fv : Coeficientes de sitio (interpolados de las tablas).
    importance : Coeficiente de importancia I.
    T : Array de periodos (s).

    Retorna
    -------
    Sa : Aceleración espectral (g).
    T0, Tc, Tl : Periodos característicos (s).

    Ecuaciones (A.2.6-1 a A.2.6-4)
    --------------------------------
      T₀ = 0.1·Av·Fv / (Aa·Fa)
      T_C = 0.48·Av·Fv / (Aa·Fa)
      T_L = 2.4·Fv

      • 0 ≤ T < T₀  :  Sa = 2.5·Aa·Fa·I·(0.4 + 0.6·T/T₀)
      • T₀ ≤ T ≤ T_C :  Sa = 2.5·Aa·Fa·I
      • T_C < T ≤ T_L :  Sa = 1.2·Av·Fv·I / T
      • T > T_L       :  Sa = 1.2·Av·Fv·T_L·I / T²
    """
    T_safe = np.maximum(T, 1e-12)  # evitar división por cero

    T0 = 0.1 * av * fv / (aa * fa)
    Tc = 0.48 * av * fv / (aa * fa)
    Tl = 2.4 * fv

    Sa_max = 2.5 * aa * fa * importance

    Sa = np.where(
        T < T0,
        Sa_max * (0.4 + 0.6 * T / T0),
        np.where(
            T <= Tc,
            Sa_max,
            np.where(
                T <= Tl,
                1.2 * av * fv * importance / T_safe,
                1.2 * av * fv * Tl * importance / T_safe**2,
            ),
        ),
    )

    return Sa, T0, Tc, Tl


def classify_zone(aa: float) -> str:
    """Clasifica la zona de amenaza sísmica según Aa."""
    if aa < 0.10:
        return "Baja"
    elif aa < 0.20:
        return "Intermedia"
    return "Alta"


def _find_regions(mask: np.ndarray):
    """Devuelve listas de (inicio, fin) de segmentos contiguos True."""
    diff = np.diff(mask.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — DATOS DE ENTRADA                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.header("Parámetros de entrada")

    # --- Coeficientes de aceleración ---
    st.subheader("Coeficientes sísmicos")
    aa = st.number_input(
        "Aa — Aceleración pico efectiva",
        min_value=0.02, max_value=0.50, value=0.25, step=0.05, format="%.2f",
        help="Coeficiente que representa la aceleración horizontal pico efectiva (Tabla A.2.3-2).",
    )
    av = st.number_input(
        "Av — Velocidad pico efectiva",
        min_value=0.02, max_value=0.50, value=0.25, step=0.05, format="%.2f",
        help="Coeficiente que representa la velocidad horizontal pico efectiva (Tabla A.2.3-2).",
    )

    # --- Zona sísmica (informativa) ---
    st.subheader("Zona de amenaza sísmica")
    zona_auto = classify_zone(aa)
    zona = st.selectbox(
        "Zona sísmica",
        ["Alta", "Intermedia", "Baja"],
        index=["Alta", "Intermedia", "Baja"].index(zona_auto),
        help="Clasificación automática según Aa. Puede ajustarla manualmente.",
    )

    # --- Perfil de suelo ---
    st.subheader("Perfil de suelo")
    soil = st.selectbox(
        "Tipo de perfil",
        list(SOIL_DESC.keys()),
        format_func=lambda x: f"Tipo {x} — {SOIL_DESC[x]}",
        index=3,
    )

    # --- Grupo de uso ---
    st.subheader("Grupo de uso")
    grupo = st.selectbox(
        "Grupo de uso de la edificación",
        list(I_FACTORS.keys()),
        index=0,
    )
    importance = I_FACTORS[grupo]

    # --- Coeficientes calculados ---
    fa = interp_coeff(aa, AA_BP, FA_TABLE[soil])
    fv = interp_coeff(av, AV_BP, FV_TABLE[soil])

    st.divider()
    st.subheader(" Parámetros calculados")

    c1, c2 = st.columns(2)
    c1.metric("Fa", f"{fa:.2f}")
    c2.metric("Fv", f"{fv:.2f}")
    c1.metric("I", f"{importance:.2f}")

    T0_val = 0.1 * av * fv / (aa * fa)
    Tc_val = 0.48 * av * fv / (aa * fa)
    Tl_val = 2.4 * fv

    c2.metric("T₀ (s)", f"{T0_val:.3f}")
    c1.metric("T_C (s)", f"{Tc_val:.3f}")
    c2.metric("T_L (s)", f"{Tl_val:.3f}")

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CÁLCULO DEL ESPECTRO DE DISEÑO                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

T = np.linspace(0, 4, 801)  # 0 a 4 s, paso 0.005 s
Sa_design, T0, Tc, Tl = compute_spectrum(aa, av, fa, fv, importance, T)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PESTAÑAS PRINCIPALES                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

tab1, tab2, tab3, tab4 = st.tabs([
    "Espectro de Diseño",
    "Comparación",
    "Tabla de Valores",
    "Metodología",
])

# ───────────────────────────── TAB 1: ESPECTRO ─────────────────────────
with tab1:
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=T, y=Sa_design,
        mode="lines",
        name="Espectro NSR-10",
        line=dict(color="#1f77b4", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(31,119,180,0.08)",
    ))

    # Líneas verticales en periodos clave
    annotations = [
        (T0, "T₀", "#2ca02c"),
        (Tc, "T_C", "#ff7f0e"),
        (Tl, "T_L", "#d62728"),
    ]
    for t_val, label, color in annotations:
        sa_at = float(np.interp(t_val, T, Sa_design))
        fig1.add_vline(x=t_val, line_dash="dash", line_color=color, opacity=0.6)
        fig1.add_annotation(
            x=t_val, y=sa_at,
            text=f"{label} = {t_val:.3f} s",
            showarrow=True, arrowhead=2, ax=40, ay=-35,
            font=dict(color=color, size=11),
        )

    fig1.update_layout(
        title=dict(text="Espectro de Diseño — NSR-10 (A.2.6)", font_size=18),
        xaxis_title="Periodo T (s)",
        yaxis_title="Aceleración espectral Sa (g)",
        template="plotly_white",
        height=560,
        hovermode="x unified",
        xaxis=dict(range=[0, 4], dtick=0.5),
        yaxis=dict(rangemode="tozero"),
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.info(
        f"**Resumen del espectro de diseño**\n\n"
        f"- Sa máx = **{Sa_design.max():.4f} g** "
        f"(meseta entre T₀ = {T0:.3f} s y T_C = {Tc:.3f} s)\n"
        f"- Fa = {fa:.2f}  ·  Fv = {fv:.2f}  ·  I = {importance:.2f}\n"
        f"- Zona sísmica: **{zona}**  ·  Perfil de suelo: **Tipo {soil}**"
    )

# ───────────────────────────── TAB 2: COMPARACIÓN ─────────────────────
with tab2:
    st.subheader("Espectro de comparación")

    input_method = st.radio(
        "Método de entrada:",
        [" Ingresar valores manualmente", " Subir archivo CSV"],
        horizontal=True,
    )

    comparison_data = None

    if input_method == " Ingresar valores manualmente":
        st.markdown(
            "Ingrese pares **(T, Sa)** en la tabla. Puede agregar o eliminar filas."
        )
        default_df = pd.DataFrame({
            "T (s)": [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
            "Sa (g)": [0.30, 0.70, 1.05, 1.10, 0.95, 0.60, 0.45, 0.30, 0.22, 0.14, 0.09],
        })

        edited_df = st.data_editor(
            default_df,
            num_rows="dynamic",
            use_container_width=True,
            key="manual_spectrum",
        )

        if not edited_df.empty:
            edited_df = edited_df.dropna()
            if len(edited_df) >= 2:
                comparison_data = (
                    edited_df.sort_values("T (s)").reset_index(drop=True)
                )

    else:  # CSV upload
        st.markdown(
            "El archivo CSV debe tener dos columnas: **T** (periodo en s) y "
            "**Sa** (aceleración en g).  Separador: coma o punto y coma."
        )
        uploaded = st.file_uploader("Subir archivo CSV", type=["csv", "txt"])

        if uploaded is not None:
            try:
                raw = uploaded.read().decode("utf-8")
                try:
                    df_up = pd.read_csv(StringIO(raw), sep=",")
                except Exception:
                    df_up = pd.read_csv(StringIO(raw), sep=";")

                df_up.columns = [c.strip() for c in df_up.columns]

                # Identificar columnas
                t_col, sa_col = None, None
                for c in df_up.columns:
                    cl = c.lower()
                    if ("t" in cl or "periodo" in cl) and "sa" not in cl:
                        t_col = c
                    elif "sa" in cl or "accel" in cl or "acel" in cl:
                        sa_col = c
                if t_col is None:
                    t_col = df_up.columns[0]
                if sa_col is None:
                    sa_col = df_up.columns[1]

                comparison_data = pd.DataFrame({
                    "T (s)": pd.to_numeric(df_up[t_col], errors="coerce"),
                    "Sa (g)": pd.to_numeric(df_up[sa_col], errors="coerce"),
                }).dropna().sort_values("T (s)").reset_index(drop=True)

                st.success(f"Archivo cargado correctamente: **{len(comparison_data)}** puntos.")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

    # ─── Graficar comparación ────────────────────────────────────────
    if comparison_data is not None and len(comparison_data) >= 2:
        T_comp = comparison_data["T (s)"].values
        Sa_comp = comparison_data["Sa (g)"].values

        # Interpolar el espectro de diseño en los puntos del usuario
        Sa_design_at_comp = np.interp(T_comp, T, Sa_design)

        # Interpolar espectro de comparación en la malla fina
        Sa_comp_interp = np.interp(T, T_comp, Sa_comp)

        # Análisis de excedencia
        exceedance_pts = Sa_comp > Sa_design_at_comp
        exceed_mask = Sa_comp_interp > Sa_design

        # ─── Gráfica ────────────────────────────────────────────────
        fig2 = go.Figure()

        # Espectro de diseño
        fig2.add_trace(go.Scatter(
            x=T, y=Sa_design,
            mode="lines", name="Espectro NSR-10",
            line=dict(color="#1f77b4", width=2.5),
        ))

        # Espectro de comparación (interpolado)
        fig2.add_trace(go.Scatter(
            x=T, y=Sa_comp_interp,
            mode="lines", name="Espectro de comparación",
            line=dict(color="#ff7f0e", width=2.5),
        ))

        # ── Relleno rojo: zonas donde comparación > diseño ──
        regions_exc = _find_regions(exceed_mask)
        for idx, (s, e) in enumerate(regions_exc):
            Tr = T[s:e]
            fig2.add_trace(go.Scatter(
                x=np.concatenate([Tr, Tr[::-1]]),
                y=np.concatenate([Sa_comp_interp[s:e], Sa_design[s:e][::-1]]),
                fill="toself",
                fillcolor="rgba(220,38,38,0.22)",
                line=dict(color="rgba(220,38,38,0)"),
                name="Excedencia",
                showlegend=(idx == 0),
                hoverinfo="skip",
            ))

        # ── Relleno verde: zonas donde diseño > comparación ──
        comply_mask = Sa_design > Sa_comp_interp
        regions_ok = _find_regions(comply_mask)
        for idx, (s, e) in enumerate(regions_ok):
            Tr = T[s:e]
            fig2.add_trace(go.Scatter(
                x=np.concatenate([Tr, Tr[::-1]]),
                y=np.concatenate([Sa_design[s:e], Sa_comp_interp[s:e][::-1]]),
                fill="toself",
                fillcolor="rgba(22,163,74,0.10)",
                line=dict(color="rgba(22,163,74,0)"),
                name="Cumple",
                showlegend=(idx == 0),
                hoverinfo="skip",
            ))

        # Puntos ingresados (color según cumplimiento)
        marker_colors = [
            "#dc2626" if exc else "#16a34a" for exc in exceedance_pts
        ]
        fig2.add_trace(go.Scatter(
            x=T_comp, y=Sa_comp,
            mode="markers",
            name="Puntos ingresados",
            marker=dict(
                color=marker_colors,
                size=9,
                line=dict(width=1, color="black"),
            ),
        ))

        fig2.update_layout(
            title=dict(text="Comparación de Espectros", font_size=18),
            xaxis_title="Periodo T (s)",
            yaxis_title="Aceleración espectral Sa (g)",
            template="plotly_white",
            height=560,
            hovermode="x unified",
            xaxis=dict(range=[0, 4], dtick=0.5),
            yaxis=dict(rangemode="tozero"),
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ─── Indicadores de cumplimiento ─────────────────────────────
        n_exceed = int(np.sum(exceedance_pts))
        n_total = len(exceedance_pts)
        pct_exceed = 100 * n_exceed / n_total if n_total else 0

        col_a, col_b, col_c = st.columns(3)

        if n_exceed == 0:
            col_a.success(
                " **CUMPLE** — El espectro de comparación no excede al "
                "espectro de diseño en ningún punto evaluado."
            )
        else:
            col_a.error(
                f" **EXCEDENCIA** en **{n_exceed}** de **{n_total}** puntos "
                f"({pct_exceed:.1f} %)"
            )

        col_b.metric("Puntos que exceden", f"{n_exceed} / {n_total}")

        max_diff = float(np.max(Sa_comp - Sa_design_at_comp))
        col_c.metric(
            "Excedencia máxima",
            f"{max_diff:.4f} g" if max_diff > 0 else "0 g",
        )

        # ─── Tabla detallada de excedencias ──────────────────────────
        if n_exceed > 0:
            st.subheader(" Detalle de excedencias por punto")
            diff_vals = Sa_comp - Sa_design_at_comp
            pct_vals = 100 * diff_vals / np.where(
                Sa_design_at_comp > 0, Sa_design_at_comp, 1e-12
            )
            exc_df = pd.DataFrame({
                "T (s)": T_comp[exceedance_pts],
                "Sa comparación (g)": Sa_comp[exceedance_pts],
                "Sa diseño (g)": Sa_design_at_comp[exceedance_pts],
                "Diferencia (g)": diff_vals[exceedance_pts],
                "Excedencia (%)": pct_vals[exceedance_pts],
            })
            st.dataframe(
                exc_df.style.format({
                    "T (s)": "{:.3f}",
                    "Sa comparación (g)": "{:.4f}",
                    "Sa diseño (g)": "{:.4f}",
                    "Diferencia (g)": "{:.4f}",
                    "Excedencia (%)": "{:.2f}",
                }),
                use_container_width=True,
            )

        # ─── Tabla completa de comparación ───────────────────────────
        with st.expander("Ver tabla completa de comparación punto a punto"):
            full_df = pd.DataFrame({
                "T (s)": T_comp,
                "Sa comparación (g)": Sa_comp,
                "Sa diseño (g)": Sa_design_at_comp,
                "Diferencia (g)": Sa_comp - Sa_design_at_comp,
                "¿Excede?": [" Sí" if e else " No" for e in exceedance_pts],
            })
            st.dataframe(
                full_df.style.format({
                    "T (s)": "{:.3f}",
                    "Sa comparación (g)": "{:.4f}",
                    "Sa diseño (g)": "{:.4f}",
                    "Diferencia (g)": "{:.4f}",
                }),
                use_container_width=True,
            )
    else:
        st.warning("Ingrese al menos **2 puntos** del espectro de comparación.")

# ───────────────────────────── TAB 3: TABLA DE VALORES ─────────────────
with tab3:
    st.subheader("Tabla de valores del espectro de diseño NSR-10")

    step = st.selectbox(
        "Paso del periodo ΔT (s):",
        [0.005, 0.01, 0.02, 0.05, 0.1],
        index=1,
    )

    T_table = np.arange(0, 4 + step / 2, step)
    Sa_table, *_ = compute_spectrum(aa, av, fa, fv, importance, T_table)

    table_df = pd.DataFrame({"T (s)": T_table, "Sa (g)": Sa_table})

    st.dataframe(
        table_df.style.format({"T (s)": "{:.3f}", "Sa (g)": "{:.4f}"}),
        use_container_width=True,
        height=500,
    )

    csv_data = table_df.to_csv(index=False)
    st.download_button(
        label=" Descargar tabla en CSV",
        data=csv_data,
        file_name=f"espectro_NSR10_Aa{aa}_Av{av}_Suelo{soil}.csv",
        mime="text/csv",
    )

# ──────────────────────────── TAB 4: METODOLOGÍA ──────────────────────
with tab4:
    st.subheader("Metodología de cálculo — NSR-10, A.2.6")

    st.markdown(r"""
### Periodos característicos

$$
T_0 = 0.1 \cdot \frac{A_v \cdot F_v}{A_a \cdot F_a}
\qquad
T_C = 0.48 \cdot \frac{A_v \cdot F_v}{A_a \cdot F_a}
\qquad
T_L = 2.4 \cdot F_v
$$

### Aceleración espectral de diseño $S_a(T)$

| Rango de periodos | Ecuación |
|---|---|
| $0 \leq T < T_0$ | $S_a = 2.5 \cdot A_a \cdot F_a \cdot I \cdot \left(0.4 + 0.6\,\dfrac{T}{T_0}\right)$ |
| $T_0 \leq T \leq T_C$ | $S_a = 2.5 \cdot A_a \cdot F_a \cdot I$ |
| $T_C < T \leq T_L$ | $S_a = \dfrac{1.2 \cdot A_v \cdot F_v \cdot I}{T}$ |
| $T > T_L$ | $S_a = \dfrac{1.2 \cdot A_v \cdot F_v \cdot T_L \cdot I}{T^2}$ |

### Coeficientes de sitio

- **$F_a$** se obtiene de la **Tabla A.2.4-3** en función del tipo de perfil de suelo
  y de $A_a$ (interpolación lineal para valores intermedios).
- **$F_v$** se obtiene de la **Tabla A.2.4-4** en función del tipo de perfil de suelo
  y de $A_v$.

### Coeficiente de importancia $I$

| Grupo de uso | I |
|---|---|
| I  – Estructuras de ocupación normal | 1.00 |
| II – Estructuras de ocupación especial | 1.10 |
| III – Edificaciones de atención a la comunidad | 1.25 |
| IV – Edificaciones indispensables | 1.50 |

### Zona de amenaza sísmica

| Zona | Condición |
|---|---|
| Baja | $A_a < 0.10$ |
| Intermedia | $0.10 \leq A_a < 0.20$ |
| Alta | $A_a \geq 0.20$ |

---

**Referencia:** *NSR-10, Título A — Requisitos generales de diseño y construcción
sismo resistente. Capítulo A.2 — Zonas de amenaza sísmica y movimientos sísmicos
de diseño.*
    """)

# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PIE DE PÁGINA                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
st.divider()
st.caption(
    "**Referencia:** NSR-10, Título A — Requisitos generales de diseño y "
    "construcción sismo resistente. Capítulo A.2.  \n"
    "*Esta herramienta es de apoyo académico y profesional. "
    "Los resultados deben verificarse contra la normativa vigente.*"
)
