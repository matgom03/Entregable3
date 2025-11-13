import dash
from dash import dcc, html,dash_table,Input,Output,State,no_update
import dash_bootstrap_components as dbc
from utils import *
import plotly.express as px
import os
import gdown
import joblib
from utils import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from sklearn.base import BaseEstimator, TransformerMixin
class MultiHotDeckImputer(BaseEstimator, TransformerMixin):
    def __init__(self, imputations, random_state=0):
        self.imputations = imputations
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        rng = np.random.default_rng(self.random_state)
        for col, group in self.imputations:
            out = X[col].copy()
            for g, sub in X.groupby(group):
                pool = sub[col].dropna().to_numpy()
                idx = sub.index[sub[col].isna()]
                if pool.size > 0 and len(idx) > 0:
                    out.loc[idx] = rng.choice(pool, size=len(idx), replace=True)
            X[col] = out
        return X
df1_head, df2_head, df = load_data()
df["Education-num"] = df["Education-num"].astype("category")
df["Income"] = df["Income"].apply(lambda x: 1 if str(x).strip().startswith(">50K") else 0)
df["Income"] = df["Income"].astype("category")
cat=df.select_dtypes(include=['object','category']).columns.drop(['Income','Education-num'])
num = df.select_dtypes(include=[np.number]).columns
duplicadas = df[df.duplicated()]
best_model = None
url = "https://drive.google.com/uc?id=1o5TPavr7g-QysNxxey9AJEA3OY95kzM2"
output_path = "modelo/gridsearch_rf.joblib"
resultados = np.load("modelo/results_rf.npz")
cm = resultados["cm"]
fpr, tpr, roc_auc = resultados["fpr"], resultados["tpr"], resultados["roc_auc"]
cm_fig = plot_confusion_matrix(cm)
roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
def load_model_lazy():
    """Carga el modelo solo cuando se necesite"""
    global best_model

    if best_model is not None:
        # Ya está cargado en memoria
        return best_model

    # Descarga solo si no existe localmente
    if not os.path.exists(output_path):
        print("Descargando modelo desde Google Drive...")
        gdown.download(url, output_path, quiet=False)

    # Carga el modelo y resultados
    print("Cargando modelo Random Forest...")
    best_model = joblib.load(output_path)
    return best_model
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)
app.title = "EDA Adult Dataset"
server = app.server
data = [
    {"Clase": "0", "Precisión": 0.93, "Recall": 0.86, "F1-score": 0.89, "Soporte": 12435},
    {"Clase": "1", "Precisión": 0.63, "Recall": 0.78, "F1-score": 0.70, "Soporte": 3846},
    {"Clase": "accuracy", "Precisión": "", "Recall": "", "F1-score": 0.84, "Soporte": 16281},
    {"Clase": "macro avg", "Precisión": 0.78, "Recall": 0.82, "F1-score": 0.79, "Soporte": 16281},
    {"Clase": "weighted avg", "Precisión": 0.86, "Recall": 0.84, "F1-score": 0.84, "Soporte": 16281},
]
subtabs_analisis = html.Div([
    dcc.Tabs(id='subtabs_eda', value='univariado', children=[
        dcc.Tab(label='1. Análisis Univariado', value='univariado'),
        dcc.Tab(label='2. Análisis Bivariado', value='bivariado'),
        dcc.Tab(label='3. Correlaciones y Colinealidad', value='correlaciones'),
        dcc.Tab(label='4. VisualizacionesModelo', value='modelo'),
        dcc.Tab(label='5. Indicadores del Modelo', value='indicadores')
    ]),
    html.Div(id='contenido_subtab_eda')
])
# ==========================================================
# CALLBACKS DE LAS SUBTABS DE ANÁLISIS (EDA)
# ==========================================================

@app.callback(
    Output('contenido_subtab_eda', 'children'),
    Input('subtabs_eda', 'value')
)
def render_subtab(subtab_value):
    # -----------------------------
    # 1 Análisis Univariado
    # -----------------------------
    if subtab_value == 'univariado':
        return html.Div([
            html.H3("Análisis Univariado del Dataset"),
            html.Div([
                html.H5("Selecciona el tipo de análisis:"),
                dcc.RadioItems(
                    id='opcion-analisis-univariado',
                    options=[
                        {'label': 'Resumen General', 'value': 'resumen'},
                        {'label': 'Variables Categóricas', 'value': 'categoricas'},
                        {'label': 'Variables Numéricas', 'value': 'numericas'},
                        {'label': 'Variable Individual', 'value': 'individual'}
                    ],
                    value='resumen',
                    inline=True,
                    labelStyle={'marginRight': '10px'}
                )
            ], style={'margin': '20px 0'}),
            html.Div(id='contenido-analisis-univariado')
        ])

    # -----------------------------
    # 2 Análisis Bivariado
    # -----------------------------
    elif subtab_value == 'bivariado':
        return html.Div([
            html.H3("Análisis Bivariado del Dataset"),

            # ================================
            # 🔹 Sección de Prueba de Normalidad
            # ================================
            html.H4("Prueba de Normalidad - Kolmogorov-Smirnov"),
            html.P("""
                Antes de realizar el análisis bivariado, se evalúa la normalidad de las variables numéricas 
                usando la prueba de Kolmogorov-Smirnov (K-S Test). Esto permite decidir entre el uso de ANOVA 
                o pruebas no paramétricas como Kruskal-Wallis.
            """),
            dbc.Button("Ejecutar prueba de normalidad", id="btn-prueba-normalidad", color="secondary", className="mb-3"),
            html.Div(id="resultado-prueba-normalidad"),
            html.Hr(),

            # ================================
            # 🔹 Sección de Análisis Bivariado
            # ================================
            html.H5("Selecciona el tipo de análisis:"),
            dcc.RadioItems(
                id='tipo_bivariado',
                options=[
                    {'label': 'Numérico vs Numérico', 'value': 'num_num'},
                    {'label': 'Categórico vs Categórico', 'value': 'cat_cat'},
                    {'label': 'Categórico vs Numérico', 'value': 'cat_num'},
                ],
                value='num_num',
                inline=True
            ),
            html.Br(),
            dbc.Button("Ejecutar análisis", id="btn_bivariado", color="primary", className="mb-3"),
            html.Div(id="salida_bivariado")
        ])


    # -----------------------------
    # 3 Correlaciones y Colinealidad
    # -----------------------------
    elif subtab_value == 'correlaciones':
        resultados_corr = analizar_colinealidad_y_correlaciones(df)
        corr_vif = resultados_corr["resultados"]["vif"]
        figs = resultados_corr["figuras"]  # antes era "imagenes"

        return html.Div([
            html.H3("Correlaciones y Colinealidad"),
            html.Hr(),

            # --------------------------
            # Tabla VIF
            # --------------------------
            html.Br(),
            html.H4("Factor de Inflación de Varianza (VIF)"),
            dash_table.DataTable(
                data=corr_vif.to_dict('records'),
                columns=[{"name": c, "id": c} for c in corr_vif.columns],
                style_table={'overflowX': 'auto', 'width': '80%', 'margin': 'auto'},
                style_header={
                    'backgroundColor': '#003366',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': '8px',
                    'fontFamily': 'Arial',
                    'fontSize': 15
                },
                style_data_conditional=[
                    {
                        'if': {'filter_query': '{VIF} > 10'},
                        'backgroundColor': '#F8D7DA',
                        'color': 'black'
                    },
                    {
                        'if': {'filter_query': '{Colinealidad} = "Alta"'},
                        'backgroundColor': '#FADBD8'
                    }
                ]
            ),

            # --------------------------
            # Gráficos Plotly
            # --------------------------
            html.Br(),
            html.H4("Visualizaciones de Correlaciones"),
            html.Div(
                [
                    dcc.Graph(figure=fig, style={"width": "48%", "display": "inline-block", "margin": "10px"})
                    for fig in figs
                ],
                style={'textAlign': 'center'}
            )
        
        ])
    # ================================
    # Subtab: Visualizaciones del Modelo
    # ================================
    elif subtab_value == 'modelo':
        model= load_model_lazy()
        fi_fig = plot_feature_importance(model, cat, num)

        return html.Div([
            html.H3("Visualizaciones del Modelo - Random Forest"),
            html.Hr(),

            html.Div([
                html.Div([
                    html.H5("Matriz de Confusión"),
                    dcc.Graph(figure=cm_fig, style={"height": "400px"})
                ], style={"width": "48%", "padding": "10px"}),

                html.Div([
                    html.H5("Curva ROC"),
                    dcc.Graph(figure=roc_fig, style={"height": "400px"})
                ], style={"width": "48%", "padding": "10px"})
            ], style={"display": "flex", "justifyContent": "space-between", "flexWrap": "wrap"}),

            html.Div([
                html.H5("Importancia de Variables"),
                dcc.Graph(figure=fi_fig, style={"height": "500px"})
            ])
        ])
    # ================================
    # Subtab: Indicadores del Modelo
    # ================================
    elif subtab_value == 'indicadores':
        # Crear DataFrame a partir de las métricas definidas
        df_metrics = pd.DataFrame(data)

        return html.Div([
            html.H3("Indicadores Globales del Modelo - Random Forest"),
            html.Hr(),

            html.H4("Reporte de Clasificación"),
            dash_table.DataTable(
                data=df_metrics.to_dict('records'),
                columns=[{"name": c, "id": c} for c in df_metrics.columns],
                style_table={'width': '80%', 'margin': 'auto'},
                style_header={
                    'backgroundColor': '#003366',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center'
                },
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontFamily': 'Arial',
                    'fontSize': 15
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'},
                    {'if': {'filter_query': '{F1-score} >= 0.85'},
                    'backgroundColor': '#D4EDDA'}
                ]
            ),

            html.Br(),
            html.P(
                "Estos indicadores reflejan el desempeño global del modelo Random Forest "
                "entrenado sobre el dataset Adult. Se observa un buen equilibrio general, "
                "con mayor desempeño en la clase 0 (ingresos ≤ 50K) y mejora del recall en la clase 1 "
                "tras aplicar SMOTE.",
                style={"textAlign": "center", "maxWidth": "80%", "margin": "auto"}
            )
    ])
    return html.Div()

@app.callback(
    Output('contenido-analisis-univariado', 'children'),
    Input('opcion-analisis-univariado', 'value')
)
def mostrar_analisis_univariado(opcion):
    if opcion == 'resumen':
        resumen_cat = analizar_categoricas(df)
        resumen_num = resumen_numericas(df)

        return html.Div([
            html.H4("Resumen General del Dataset"),
            html.H5("Variables Categóricas"),
            dash_table.DataTable(
                data=resumen_cat.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen_cat.columns],
                page_size=10,
                style_table={"overflowX": "auto"}
            ),
            html.Br(),
            html.H5("Variables Numéricas"),
            dash_table.DataTable(
                data=resumen_num.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen_num.columns],
                page_size=10,
                style_table={"overflowX": "auto"}
            )
        ])

    elif opcion == 'categoricas':
        resumen_cat = analizar_categoricas(df)
        graficos_barra = graficar_categoricas(df, tipo="barra")

        return html.Div([
            html.H4("Análisis de Variables Categóricas"),
            dash_table.DataTable(
                data=resumen_cat.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen_cat.columns],
                page_size=10,
                style_table={"overflowX": "auto"}
            ),
            html.Br(),
            html.H5("Gráficos de Frecuencia"),
            html.Div([html.Img(src=img, style={"width": "40%", "margin": "10px"}) for img in graficos_barra])
        ])

    elif opcion == 'numericas':
        resumen_num = resumen_numericas(df)
        graficos_num = graficar_numericas(df)

        return html.Div([
            html.H4("Análisis de Variables Numéricas"),
            dash_table.DataTable(
                data=resumen_num.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen_num.columns],
                page_size=10,
                style_table={"overflowX": "auto"}
            ),
            html.Br(),
            html.H5("Distribución Numérica"),
            html.Div([html.Img(src=img, style={"width": "45%", "margin": "10px"}) for img in graficos_num])
        ])

    elif opcion == 'individual':
        return html.Div([
            html.H4("Análisis de Variable Individual"),
            dcc.Dropdown(
                id='variable-individual',
                options=[{'label': c, 'value': c} for c in df.columns],
                value=df.columns[0],
                style={'width': '50%'}
            ),
            html.Div(id='analisis-variable-univariado')
        ])

    return html.Div("Selecciona una opción de análisis.")
@app.callback(
    Output('analisis-variable-univariado', 'children'),
    Input('variable-individual', 'value')
)
def analizar_variable_individual(columna):
    if columna is None:
        return html.Div("Selecciona una variable para analizar.")

    if pd.api.types.is_numeric_dtype(df[columna]):
        resumen = df[[columna]].describe().T
        mediana = df[columna].median()
        varianza = df[columna].var()

        fig_hist = px.histogram(
            df, x=columna, nbins=20, title=f"Histograma de {columna}",
            marginal="box", labels={columna: columna}
        )

        return html.Div([
            html.H5(f"Variable seleccionada: {columna}"),
            dash_table.DataTable(
                data=resumen.to_dict('records'),
                columns=[{"name": i, "id": i} for i in resumen.columns]
            ),
            html.P(f"Mediana: {mediana:.2f}"),
            html.P(f"Varianza: {varianza:.2f}"),
            dcc.Graph(figure=fig_hist)
        ])

    else:
        conteo = df[columna].value_counts(dropna=False).reset_index()
        conteo.columns = [columna, "Frecuencia"]

        fig_bar = px.bar(
            conteo, x=columna, y="Frecuencia",
            title=f"Frecuencia de {columna}", text="Frecuencia"
        )

        children = [
            html.H5(f"Variable seleccionada: {columna}"),
            dash_table.DataTable(
                data=conteo.to_dict('records'),
                columns=[{"name": i, "id": i} for i in conteo.columns]
            ),
            dcc.Graph(figure=fig_bar)
        ]

        
        if conteo.shape[0] <= 6:
            fig_pie = px.pie(conteo, values="Frecuencia", names=columna,
                             title=f"Distribución de {columna}")
            children.append(dcc.Graph(figure=fig_pie))
        else:
            children.append(html.P(" Demasiadas categorías para gráfico de pastel."))

        return html.Div(children)
@app.callback(
    Output("resultado-prueba-normalidad", "children"),
    Input("btn-prueba-normalidad", "n_clicks"),
    prevent_initial_call=True
)
def ejecutar_prueba_normalidad(n_clicks):
    if n_clicks == 0:
        return no_update
    resultado = prueba_normalidad(df)
    if isinstance(resultado, pd.DataFrame):
        return dash_table.DataTable(
            data=resultado.to_dict("records"),
            columns=[{"name": c, "id": c} for c in resultado.columns],
            page_size=10,
            style_table={"overflowX": "auto"}
        )
    else:
        return html.Pre(str(resultado))
@app.callback(
    Output("salida_bivariado", "children"),
    Input("btn_bivariado", "n_clicks"),
    State("tipo_bivariado", "value"),
    prevent_initial_call=True
)
def mostrar_bivariado(n_clicks, tipo):
    if tipo == 'num_num':
        resultado = analisis_bivariado_numerico(df)
    elif tipo == 'cat_cat':
        resultado = analisis_bivariado_categorico(df)
    else:
        resultado = analisis_bivariado_cat_num(df)

    if "error" in resultado:
        return html.Div(resultado["error"])

    tabla = dash_table.DataTable(
        data=resultado["tabla"].round(4).to_dict('records'),
        columns=[{"name": c, "id": c} for c in resultado["tabla"].columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
    )

    graficos = [dcc.Graph(figure=fig) for fig in resultado["imagenes"]]

    return html.Div([tabla] + graficos)

tab_introduccion=dcc.Tab(label='1. Introducción', children=[
        html.H2('Exploratory Data Analysis - Adult Dataset'),
        html.P("""
            Bienvenido al panel interactivo del Análisis Exploratorio de Datos (EDA)
            para el dataset Adult / Census Income del repositorio UCI.
        """),
        html.P("""
            Este tablero permite explorar el conjunto de datos, comprender su estructura,
            analizar distribuciones, relaciones entre variables y detectar posibles problemas
            antes del modelado predictivo. Una vez realizado esto se elabora un modelo de machine Learning en el que se le analizan sus resultados y graficas.
        """)
    ])
tab_contexto = dbc.Tab(
    label="2. Contexto",
    children=[
        html.Div(
            [
                html.H2("Contexto del EDA - Adult Dataset"),
                html.Hr(),
                dcc.Markdown(
                    """
                    ## Contexto del dataset

                    En el presente trabajo se realiza el análisis del dataset **Adult / Census Income**
                    del **UCI Machine Learning Repository**.  
                    Publicado por Barry Becker en 1996, con datos del censo de EE. UU. de 1994.  
                    Contiene **48,842 instancias** y **14 atributos**.

                    **Fuente:** [https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult)

                    ---

                    ## Problema
                    Predecir si el **ingreso anual de una persona es mayor o menor a 50,000 USD**,
                    basándose en características demográficas, educativas y laborales.

                    ---

                    ## Objetivos
                    - Limpiar y transformar el dataset original.  
                    - Analizar variables numéricas y categóricas.  
                    - Identificar relaciones entre educación, ocupación y nivel de ingresos.  
                    - Preparar datos para modelado predictivo.
                    - Realizar modelado a traves del modelo de RandomForest y evaluar su desempeño

                    ---

                    ## Variables incluidas
                    - Age  
                    - Fnlwgt  
                    - Education  
                    - Education-num  
                    - Capital-gain  
                    - Capital-loss  
                    - Hours-per-week  
                    - Workclass  
                    - Occupation  
                    - Native-country  
                    - Relationship  
                    - Marital-status  
                    - Race  
                    - Sex  
                    - Income

                    ---

                    ## Alcance
                    Este análisis es descriptivo, enfocado en la exploración de patrones y relaciones. Luego se evaluan las metricas y graficas de la implementacion de un modelo de randomforest. 
                    """
                ),
            ],
            style={"padding": "20px"},
        )
    ],
)
tab_etl = dbc.Tab(
    label="3. ETL - Extracción, Transformación y Carga",
    children=[
        html.Div(
            [
                html.H2("ETL - Extracción, Transformación y Carga, Análisis inicial"),
                html.Hr(),
                html.P(
                    "Se realizó la concatenación de los dos dataframes (train y test) para crear un dataset general, "
                    "agregando también los nombres de las columnas."
                ),

                html.H4("Vista previa de 'adult.data'"),
                dash_table.DataTable(
                    data=df1_head.to_dict("records"),
                    columns=[{"name": str(i), "id": str(i)} for i in df1_head.columns],
                    page_size=5,
                    style_table={"overflowX": "auto"},
                ),

                html.H4("Vista previa de 'adult.test'"),
                dash_table.DataTable(
                    data=df2_head.to_dict("records"),
                    columns=[{"name": str(i), "id": str(i)} for i in df2_head.columns],
                    page_size=5,
                    style_table={"overflowX": "auto"},
                ),
                html.Hr(),
                html.H4("Dataset final concatenado"),
                dash_table.DataTable(
                    data=df.head().to_dict("records"),
                    columns=[{"name": str(i), "id": str(i)} for i in df.columns],
                    page_size=5,
                    style_table={"overflowX": "auto"},
                ),

                html.Br(),
                html.H4("Tipos de datos despues de concatenar y transformar"),
                html.Pre(str(df.dtypes)),

                html.Hr(),
                html.H4("Filas duplicadas"),
                html.P(f"Total de filas duplicadas: {len(duplicadas)}"),

                (
                    dash_table.DataTable(
                        data=duplicadas.to_dict("records"),
                        columns=[{"name": str(i), "id": str(i)} for i in duplicadas.columns],
                        page_size=5,
                        style_table={"overflowX": "auto"},
                    )
                    if len(duplicadas) > 0
                    else html.Div(
                        dbc.Alert("No se encontraron filas duplicadas.", color="success")
                    )
                ),

                html.P(
                    "En general, estas filas duplicadas se deben más a casualidad que a error, "
                    "por lo que no se van a eliminar."
                ),
            ],
            style={"padding": "20px"},
        )
    ],
)
tab_Resultados = dbc.Tab(
    label='4. Resultados y Analisis Final',
    children=[
        html.Div([
            html.H2('Análisis Exploratorio de Datos (EDA) y Modelo'),
            html.Hr(),
            subtabs_analisis
        ], style={"padding": "20px"})
    ]
)
tab_conclusiones = dbc.Tab(
    label="5. Conclusiones",
    children=[
        html.Div(
            [
                html.H2("Conclusiones del Análisis Exploratorio de Datos (EDA)"),
                html.Hr(),
                dcc.Markdown(
                    """
                    ## Conclusiones generales sobre analisis univariado 
                    - Se observa la presencia de datos atipicos para las variables numericas y presencia de datos Nan para las variables categoricas que deben ser tratados
                    - Se observa que algunos de los graficos como el de Age o Fnlwgt tienen un sesgo positivo, por lo que la mayoria de los datos se concentran en los valores inferiores 
                    - Se observa tambien que hay bastantes valores de 0 en capital gain o capital loss, sin embargo estos valores no los podemos considerar como valores faltantes ya que si son valores reales
                    - Tambien en las variables categoricas se observa que la variable income esta desbalanceada, esto se debe tener en cuenta si se realiza algun modelo con esta variable como objetivo


                    ---

                    ## Conclusiones generales sobre analisis bivariado 
                    - Podemos observar que en general, las correlaciones entre las variables numericas del dataset son bastante bajas, tanto en las correlaciones de pearson y spearman ninguno supera mas de 0.2 de correlacion. 
                    En general al realizar la correlacion con el metodo de spearman se generaron mejores resultados de los que habia a comparación de cuando se calcula con spearman
                    - Podemos observar que en las variables categoricas, el p-valor es menor que 0.05 por lo que todos los pares tienen relaciones significativas. 
                    Sin embargo para un mayor entendimiento se calculo tambien la fuerza mediante cramer para ver cuales eran asociadas con mayor fuerza, las cuales en su mayoria son las variables que estaban mas relacionadas en concepto como raza y pais nativo, las mas moderadas tampoco se pueden ignorar porque estas tambien muestran patrones importantes como la relacion entre ingresos y la educacion 
                    - Se utilizaron las pruebas del test de levene y la normalidad con kstest para determinar si se debia usar ANOVA o Kruskall wallis para el analisis bivariado entre categoricas y numericas, en general al no ser normales y los pares no tener varianzas homogeneas se utilizo kruskall wallis y se nota que en la mayoria de los casos hay diferencias estadisticamente significativas entre los pares, 
                    lo que significa que hay relaciones fuertes entre ellas, la unica excepcion es  income y Fnlwgt cuyo p-valor es mayor con 0.09 por lo que los ingresos no estan relacionados con el peso de muestra poblacional




                    ---
                    ## Conclusiones generales sobre las correlaciones del dataset
                    - Podemos observar que en cuanto a las correlaciones de variables hay varias cosas curiosas por ejemplo Education y Education num tienen correlacion perfecta, por lo que si se llega a realizar algun modelado con el dataset se debe eliminar 1 de ellas para no generar data leakage.
                    - En general las variables categoricas estan muchisimo mas relacionadas que las variables numericas lo cual muestra que las variables numericas son mas independientes entre si en comparacion y se debe tomar en cuenta si se realiza el modelado con este dataset 
                    ---
                    ## Conclusiones del modelo
                    - El modelo realizado tiene una ROC AUC de 0.91 lo que nos muestra que realiza una buena discriminacion entre ambas clases 
                    - El modelo en general reconoce la mayoria de los casos de ambas clases con pocos falsos negativos en la matriz de confusion 
                    - Las metricas generales del modelo rondan alrededor de 0.7 en la clase de ganancias mayores a 50k y 0.8 para la clase de ganancias 0 
                    - Las variables mas importantes para el modelo son Las variables más importantes son edad, estado civil, horas trabajadas y capital-gain, todas intuitivamente relacionadas con ingresos.
                    - Limitaciones del modelo: el modelo al ser randomforest es bastante tardado en la ejecucion sobre todo en la busqueda de hiperparametros realizada al ser un dataset de tamaño mediano, ademas de que tiene cierto desbalance el dataset lo cual puede complicar los resultados si no se usa tecnicas de balanceo como SMOTE o ADASYN. 
                    También existen correlaciones entre variables que pueden distorsionar la interpretación de la importancia de las características, por lo que se deben tomar en cuenta estos aspectos si se quieren realizar analisis futuros.
                    """
                ),
            ],
            style={"padding": "20px"},
        )
    ],
)

tabs = dcc.Tabs([tab_introduccion,tab_contexto,tab_etl,tab_Resultados,tab_conclusiones])
app.layout = dbc.Container([
    html.H1(" Exploratory Data Analysis - Adult Dataset", className="text-center my-4"),
    html.Hr(),
    tabs
], fluid=True)
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)