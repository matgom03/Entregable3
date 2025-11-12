import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, normaltest, kstest,f_oneway, kruskal, chi2_contingency, levene,mannwhitneyu
from itertools import combinations
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
import plotly.graph_objects as go
import io
import base64
# =====================
# Función auxiliar para graficar
# =====================
def fig_to_uri(fig):
    """Convierte una figura de Matplotlib a URI para mostrar en Dash."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return "data:image/png;base64,{}".format(encoded)


# =====================
# Carga de datos
# =====================
def load_data():
    df1 = pd.read_csv("adult.data", header=None, na_values=["?", " ?"], skipinitialspace=True)
    df2 = pd.read_csv("adult.test", header=None, na_values=["?", " ?"], skipinitialspace=True, skiprows=1)
    df = pd.concat([df1, df2], ignore_index=True)
    cols = [
        "Age", "Workclass", "Fnlwgt", "Education", "Education-num",
        "Marital-status", "Occupation", "Relationship", "Race", "Sex",
        "Capital-gain", "Capital-loss", "Hours-per-week", "Native-country", "Income"
    ]
    df.columns = cols
    
    return df1.head(), df2.head(), df


# =====================
# Análisis categóricas
# =====================
def analizar_categoricas(df):
    resultados = []
    categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in categoricas:
        conteo = df[col].value_counts(dropna=False)
        proporcion = df[col].value_counts(normalize=True, dropna=False)
        nulos = df[col].isna().sum()
        categoria_moda = conteo.index[0] if not conteo.empty else None
        frecuencia_moda = conteo.iloc[0] if not conteo.empty else None
        resultados.append({
            "variable": col,
            "cantidad_categorias": df[col].nunique(dropna=False),
            "categoria_mas_frecuente": categoria_moda,
            "frecuencia_mas_frecuente": frecuencia_moda,
            "porcentaje_mas_frecuente": proporcion.iloc[0] if not conteo.empty else None,
            "nulos": nulos
        })
    return pd.DataFrame(resultados).sort_values(by="cantidad_categorias", ascending=False)


def graficar_categoricas(df, tipo="barra"):
    figs = []
    categoricas = df.select_dtypes(include=['object', 'category', 'bool']).columns
    for col in categoricas:
        conteo = df[col].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        if tipo == "barra":
            conteo.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")
            ax.set_title(f"Frecuencia de {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
        elif tipo == "pastel":
            conteo.plot(kind="pie", autopct='%1.1f%%', startangle=90, ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Distribución de {col}")
        figs.append(fig_to_uri(fig))
    return figs


# =====================
# Análisis numéricas
# =====================
def resumen_numericas(df):
    numericas = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numericas) == 0:
        return None
    resumen = df[numericas].describe().T
    resumen["varianza"] = df[numericas].var()
    resumen["mediana"] = df[numericas].median()
    return resumen


def graficar_numericas(df):
    figs = []
    numericas = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numericas:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df[col].hist(bins=20, color="skyblue", edgecolor="black", ax=axes[0])
        axes[0].set_title(f"Histograma de {col}")
        axes[1].boxplot(df[col].dropna(), vert=False, patch_artist=True,
                        boxprops=dict(facecolor="lightgreen", color="black"),
                        medianprops=dict(color="red"))
        axes[1].set_title(f"Boxplot de {col}")
        figs.append(fig_to_uri(fig))
    return figs


# =====================
# Detección de outliers
# =====================
def detectar_outliers(df):
    resultados = []
    numericas = df.select_dtypes(include=[np.number]).columns
    for col in numericas:
        serie = df[col].dropna()
        if len(serie) == 0:
            continue
        q1, q3 = serie.quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((serie < lim_inf) | (serie > lim_sup)).sum()
        total = len(serie)
        resultados.append({
            "variable": col,
            "n_muestras": total,
            "outliers": outliers,
            "proporcion_outliers": outliers / total if total > 0 else np.nan
        })
    return pd.DataFrame(resultados).sort_values(by="proporcion_outliers", ascending=False)

# ====================================================
# BIVARIADO NUMÉRICO vs NUMÉRICO (Plotly)
# ====================================================
def analisis_bivariado_numerico(df, top=5):
    num_df = df.select_dtypes(include=[np.number])
    cols = num_df.columns
    resultados = []

    if len(cols) < 2:
        return {"error": "Se necesitan al menos dos variables numéricas."}

    for var1, var2 in combinations(cols, 2):
        pearson = num_df[var1].corr(num_df[var2], method='pearson')
        spearman = num_df[var1].corr(num_df[var2], method='spearman')
        covarianza = np.cov(num_df[var1].dropna(), num_df[var2].dropna())[0, 1]
        resultados.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Correlación Pearson': pearson,
            'Correlación Spearman': spearman,
            'Covarianza': covarianza
        })

    corr_df = pd.DataFrame(resultados).sort_values(by='Correlación Spearman', ascending=False)
    figs = []
    pares_interes = pd.concat([corr_df.head(top), corr_df.tail(top)])

    for _, row in pares_interes.iterrows():
        var1, var2 = row['Variable 1'], row['Variable 2']
        fig = px.scatter(num_df, x=var1, y=var2,
                         trendline="ols",
                         title=f"{var1} vs {var2}<br>Spearman: {row['Correlación Spearman']:.3f}")
        figs.append(fig)

    return {"tabla": corr_df, "imagenes": figs}
# ============================================
# V de Cramer (para categóricas)
# ============================================
def cramers_v(x, y):
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
def Cramers_v(ct):
    chi2, _, _, _ = chi2_contingency(ct, correction=False)
    n = ct.to_numpy().sum()
    k = min(ct.shape)-1
    return np.sqrt(chi2 / (n * k)) if n > 0 and k > 0 else np.nan
# ====================================================
# BIVARIADO CATEGÓRICO vs CATEGÓRICO (Plotly)
# ====================================================
def analisis_bivariado_categorico(df, top=5):
    cat_df = df.select_dtypes(include=['object', 'category'])
    cols = cat_df.columns
    resultados = []

    if len(cols) < 2:
        return {"error": "Se necesitan al menos dos variables categóricas."}

    for var1, var2 in combinations(cols, 2):
        tabla = pd.crosstab(df[var1], df[var2])
        if tabla.shape[0] < 2 or tabla.shape[1] < 2:
            continue
        chi2, p, dof, expected = chi2_contingency(tabla)
        n = tabla.sum().sum()
        phi2 = chi2 / n
        r, k = tabla.shape
        phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
        rcorr = r - ((r - 1)**2) / (n - 1)
        kcorr = k - ((k - 1)**2) / (n - 1)
        cramer_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if min(kcorr, rcorr) > 1 else 0

        resultados.append({
            'Variable 1': var1,
            'Variable 2': var2,
            'Chi-cuadrado': chi2,
            'Grados de libertad': dof,
            'Valor p': p,
            'V de Cramer': cramer_v
        })

    resumen_df = pd.DataFrame(resultados).sort_values(by='Valor p', ascending=True).reset_index(drop=True)
    figs = []

    top_pairs = resumen_df.head(top)
    for _, row in top_pairs.iterrows():
        var1, var2 = row['Variable 1'], row['Variable 2']
        tabla_rel = pd.crosstab(df[var1], df[var2], normalize='index')
        fig = px.imshow(tabla_rel,
                        text_auto=True,
                        color_continuous_scale='YlGnBu',
                        title=f"{var1} vs {var2}<br>(p = {row['Valor p']:.4f}, V = {row['V de Cramer']:.3f})")
        figs.append(fig)

    return {"tabla": resumen_df, "imagenes": figs}
# ====================================================
# BIVARIADO CATEGÓRICO vs NUMÉRICO (Plotly)
# ====================================================
def analisis_bivariado_cat_num(df):
    categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    resultados = []
    figs = []

    for cat in categoricas:
        for num in numericas:
            grupos = [df[num][df[cat] == nivel].dropna() for nivel in df[cat].dropna().unique()]
            grupos = [g for g in grupos if len(g) > 1]
            if len(grupos) < 2:
                continue

            normalidades = []
            for g in grupos:
                g_std = (g - np.mean(g)) / np.std(g, ddof=1)
                _, p_norm = kstest(g_std, 'norm')
                normalidades.append(p_norm > 0.05)

            stat_lev, p_levene = levene(*grupos)
            homocedasticas = p_levene > 0.05

            if all(normalidades) and homocedasticas:
                prueba = "ANOVA"
                stat, p_valor = f_oneway(*grupos)
            else:
                prueba = "Kruskal-Wallis"
                stat, p_valor = kruskal(*grupos)

            resultados.append({
                'Categórica': cat,
                'Numérica': num,
                'Prueba': prueba,
                'Normalidad_OK': all(normalidades),
                'Homocedasticidad_OK': homocedasticas,
                'p-Levene': p_levene,
                'Estadístico': stat,
                'p-valor': p_valor
            })

            fig = px.box(df, x=cat, y=num, color=cat,
                         title=f'{num} por {cat}<br>({prueba}, p = {p_valor:.4f})',
                         points='all')
            figs.append(fig)

    resultados_df = pd.DataFrame(resultados).sort_values(by='p-valor') if resultados else pd.DataFrame()
    return {"tabla": resultados_df, "imagenes": figs}
def prueba_normalidad(df, alpha=0.05):
    resultados = []

    # Detectar variables numéricas
    numericas = df.select_dtypes(include=[np.number]).columns

    for col in numericas:
        serie = df[col].dropna()

        if len(serie) < 8:
            resultados.append({
                "variable": col,
                "n_muestras": len(serie),
                "shapiro_p": np.nan,
                "dagostino_p": np.nan,
                "ks_p": np.nan,
                f"es_normal (alpha={alpha})": "Muestra insuficiente"
            })
            continue

        # Shapiro-Wilk
        stat_shapiro, p_shapiro = shapiro(serie) if len(serie) <= 5000 else (np.nan, np.nan)

        # D’Agostino y Pearson
        stat_dagostino, p_dagostino = normaltest(serie)

        # Kolmogorov-Smirnov
        stat_ks, p_ks = kstest(
            (serie - serie.mean()) / serie.std(ddof=0), 'norm'
        )

        # Veredicto
        es_normal = (
            (np.isnan(p_shapiro) or p_shapiro > alpha) and
            (p_dagostino > alpha) and
            (p_ks > alpha)
        )

        resultados.append({
            "variable": col,
            "n_muestras": len(serie),
            "shapiro_p": p_shapiro,
            "dagostino_p": p_dagostino,
            "ks_p": p_ks,
            f"es_normal (alpha={alpha})": es_normal
        })

    return pd.DataFrame(resultados).sort_values(by=f"es_normal (alpha={alpha})", ascending=False)
def analizar_colinealidad_y_correlaciones(df, umbral_vif=10, mostrar_graficos=True):
    resultados = {}
    figs = []

    # Separar variables numéricas y categóricas
    num_vars = df.select_dtypes(include=['int64', 'float64']).columns
    cat_vars = df.select_dtypes(include=['object', 'category']).columns

    # ==========================
    # Correlaciones numéricas
    # ==========================
    if len(num_vars) > 1:
        corr_pearson = df[num_vars].corr(method='pearson')
        corr_spearman = df[num_vars].corr(method='spearman')
        resultados['corr_pearson'] = corr_pearson
        resultados['corr_spearman'] = corr_spearman

        if mostrar_graficos:
            # Pearson
            fig_pearson = px.imshow(
                corr_pearson,
                text_auto=".2f",
                color_continuous_scale='RdBu',
                origin='lower',
                title="Correlación Numérica (Pearson)"
            )
            fig_pearson.update_layout(height=600, width=900)
            figs.append(fig_pearson)

            # Spearman
            fig_spearman = px.imshow(
                corr_spearman,
                text_auto=".2f",
                color_continuous_scale='Tealrose',
                origin='lower',
                title="Correlación Numérica (Spearman)"
            )
            fig_spearman.update_layout(height=600, width=900)
            figs.append(fig_spearman)

        # ==========================
        # VIF
        # ==========================
        X = df[num_vars].dropna()
        vif_data = pd.DataFrame({
            "Variable": X.columns,
            "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        })
        vif_data["Colinealidad"] = np.where(vif_data["VIF"] > umbral_vif, "Alta", "Aceptable")
        resultados['vif'] = vif_data

    # ==========================
    # Correlaciones categóricas (Cramér’s V)
    # ==========================
    if len(cat_vars) > 1:
        matriz_cramer = pd.DataFrame(np.ones((len(cat_vars), len(cat_vars))), index=cat_vars, columns=cat_vars)
        for var1, var2 in combinations(cat_vars, 2):
            val = cramers_v(df[var1], df[var2])
            matriz_cramer.loc[var1, var2] = val
            matriz_cramer.loc[var2, var1] = val
        resultados['cramers_v'] = matriz_cramer

        if mostrar_graficos:
            fig_cramer = px.imshow(
                matriz_cramer,
                text_auto=".2f",
                color_continuous_scale='YlGnBu',
                origin='lower',
                title="Correlación Categórica (Cramér’s V)"
            )
            fig_cramer.update_layout(height=600, width=900)
            figs.append(fig_cramer)

    return {"resultados": resultados, "figuras": figs}
def winsorizar_outliers(df, variables=None, lower_percentile=0.01, upper_percentile=0.99, mostrar_resumen=True):
    if variables is None:
        variables = ['Capital-gain', 'Capital-loss', 'Hours-per-week']

    df_wins = df.copy()
    resumen = {}

    for var in variables:
        if var not in df.columns:
            continue

        lower = df[var].quantile(lower_percentile)
        upper = df[var].quantile(upper_percentile)
        df_wins[var] = np.clip(df[var], lower, upper)
        resumen[var] = {
            'min_original': df[var].min(),
            'max_original': df[var].max(),
            'lower_limit': lower,
            'upper_limit': upper,
            'min_winsorizado': df_wins[var].min(),
            'max_winsorizado': df_wins[var].max()
        }

    resumen_df = pd.DataFrame(resumen).T if mostrar_resumen else None
    return {"df_wins": df_wins, "resumen": resumen_df}
def hot_deck_group(df, col, group, random_state=0):
    rng = np.random.default_rng(random_state)
    out = df[col].copy()
    for g, sub in df.groupby(group):
        pool = sub[col].dropna().to_numpy()
        idx = sub.index[sub[col].isna()]
        if pool.size > 0 and len(idx) > 0:
            out.loc[idx] = rng.choice(pool, size=len(idx), replace=True)
    return out
def plot_confusion_matrix(cm):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicción", y="Real", color="Cantidad"))
    fig.update_layout(title="Matriz de Confusión")
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatorio', line=dict(dash='dash')))
    fig.update_layout(title=f"Curva ROC (AUC = {roc_auc:.2f})",
                      xaxis_title="Tasa Falsos Positivos",
                      yaxis_title="Tasa Verdaderos Positivos")
    return fig

def plot_feature_importance(modelo, cat=None, num=None):
    # --- 1. Obtener importancias del modelo ---
    if hasattr(modelo, 'named_steps'):  # Si es un pipeline
        model_final = modelo.named_steps['modelo']
        importances = model_final.feature_importances_

        # --- 2. Obtener nombres de variables ---
        try:
            encoder = modelo.named_steps['preprocesador'].named_transformers_['cat'].named_steps['onehot']
            features_cat = encoder.get_feature_names_out(cat)
        except Exception:
            features_cat = np.array([])

        features_num = np.array(num) if num is not None else np.array([])
        features = np.concatenate([features_num, features_cat])

    else:  # Si el objeto es directamente un modelo (no pipeline)
        model_final = modelo
        importances = model_final.feature_importances_

        # Si no hay nombres de variables, generar nombres genéricos
        n_features = len(importances)
        if num is not None or cat is not None:
            features = np.array((num or []) + (cat or []))
        else:
            features = np.array([f"Feature {i}" for i in range(n_features)])

    # --- 3. Crear DataFrame de importancias ---
    df_imp = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # --- 4. Graficar ---
    fig = px.bar(
        df_imp.head(15),
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 15 Importancias de Variables"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))

    return fig
