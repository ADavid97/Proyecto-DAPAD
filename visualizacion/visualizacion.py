import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from chart_theme import PALETTE


class Visualizacion:
    """Gráficas Plotly del dataset y de los modelos, con el template 'daad'."""

    def __init__(self, datos: pd.DataFrame):
        self.datos = datos

    def histograma(self, columna: str) -> go.Figure:
        fig = px.histogram(
            self.datos, x=columna,
            template="daad",
            color_discrete_sequence=PALETTE,
        )
        fig.update_layout(bargap=0.06, xaxis_title=columna, yaxis_title="Frecuencia")
        return fig

    def histograma_comparado(self, columna: str, grupo: str) -> go.Figure:
        """Histograma de `columna` superpuesto por `grupo`, para comparar datasets."""
        fig = px.histogram(
            self.datos, x=columna, color=grupo,
            template="daad",
            color_discrete_sequence=PALETTE,
            barmode="overlay",
            opacity=0.65,
        )
        # Leyenda horizontal arriba del gráfico: los nombres de origen (que
        # pueden ser largos) se ven completos en vez de cortarse a la derecha.
        fig.update_layout(
            bargap=0.06, xaxis_title=columna, yaxis_title="Frecuencia",
            legend=dict(
                title=None,
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="left", x=0,
            ),
            margin=dict(t=64),
        )
        return fig

    def boxplot(self, columna: str) -> go.Figure:
        fig = px.box(
            self.datos, y=columna,
            template="daad",
            color_discrete_sequence=PALETTE,
        )
        return fig

    def scatter(self, col_x: str, col_y: str, color: str | None = None) -> go.Figure:
        fig = px.scatter(
            self.datos, x=col_x, y=col_y, color=color,
            template="daad",
            color_discrete_sequence=PALETTE,
        )
        return fig

    def heatmap_correlacion(self) -> go.Figure:
        corr = self.datos.select_dtypes(include="number").corr().round(2)
        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale=[[0, "#2a6fbb"], [0.5, "#fafaf7"], [1, "#d2502a"]],
                zmid=0,
                text=corr.values,
                texttemplate="%{text:.2f}",
                hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>",
            )
        )
        fig.update_layout(template="daad")
        return fig

    def grafica_barras(self, col_x: str, col_y: str | None = None) -> go.Figure:
        if col_y:
            fig = px.bar(
                self.datos, x=col_x, y=col_y,
                template="daad",
                color_discrete_sequence=PALETTE,
            )
        else:
            conteo = self.datos[col_x].value_counts().reset_index()
            conteo.columns = [col_x, "conteo"]
            fig = px.bar(
                conteo, x=col_x, y="conteo",
                template="daad",
                color_discrete_sequence=PALETTE,
            )
            fig.update_layout(yaxis_title="Frecuencia")
        return fig

    def grafica_lineas(self, col_x: str, col_y: str) -> go.Figure:
        # Ordenar por X evita el zigzag cuando los datos no vienen ordenados
        datos = self.datos.sort_values(col_x)
        fig = px.line(
            datos, x=col_x, y=col_y,
            template="daad",
            color_discrete_sequence=PALETTE,
        )
        return fig

    def grafica_pastel(self, columna: str, max_categorias: int = 50) -> go.Figure:
        conteo = self.datos[columna].value_counts().reset_index()
        conteo.columns = [columna, "conteo"]
        if len(conteo) > max_categorias:
            otros = conteo.iloc[max_categorias:]["conteo"].sum()
            conteo = conteo.iloc[:max_categorias]
            conteo = pd.concat(
                [conteo, pd.DataFrame([{columna: "Otros", "conteo": otros}])],
                ignore_index=True,
            )
        fig = px.pie(
            conteo, names=columna, values="conteo",
            template="daad",
            color_discrete_sequence=PALETTE,
        )
        fig.update_traces(
            textfont_family="'IBM Plex Mono', monospace",
            textfont_size=11,
        )
        return fig

    # ── Charts used by Models / Evaluación ────────────────────────────────

    def grafica_codo(self, inercias: list) -> go.Figure:
        k_vals = list(range(1, len(inercias) + 1))
        fig = go.Figure(
            go.Scatter(
                x=k_vals, y=inercias,
                mode="lines+markers",
                line=dict(color=PALETTE[0], width=2),
                marker=dict(size=7, color=PALETTE[0]),
                name="Inercia",
            )
        )
        fig.update_layout(
            template="daad",
            xaxis_title="Número de clusters (k)",
            yaxis_title="Inercia",
        )
        return fig

    def grafica_clusters(
        self,
        datos: pd.DataFrame,
        etiquetas: np.ndarray,
        centroides: np.ndarray,
    ) -> go.Figure:
        cols = datos.select_dtypes(include="number").columns.tolist()
        two_d = len(cols) >= 2
        col_x = cols[0]
        if two_d:
            col_y = cols[1]
            df_plot = datos[[col_x, col_y]].copy()
        else:
            # Con una sola feature se grafica contra el índice de la fila
            col_y = "índice"
            df_plot = pd.DataFrame({col_x: datos[col_x].to_numpy(), col_y: np.arange(len(datos))})
        df_plot["Cluster"] = etiquetas.astype(str)
        fig = px.scatter(
            df_plot, x=col_x, y=col_y, color="Cluster",
            template="daad",
            color_discrete_sequence=PALETTE,
        )
        cx = centroides[:, 0] if centroides.ndim > 1 else centroides
        if two_d and centroides.ndim > 1 and centroides.shape[1] > 1:
            cy = centroides[:, 1]
        else:
            cy = np.full(len(cx), len(df_plot) / 2)
        fig.add_scatter(
            x=cx, y=cy,
            mode="markers",
            marker=dict(symbol="x", size=14, color="#1a1a1a", line=dict(width=2)),
            name="Centroides",
        )
        return fig

    def grafica_regresion(self, y_real: np.ndarray, y_predicho: np.ndarray) -> go.Figure:
        mn = float(min(y_real.min(), y_predicho.min()))
        mx = float(max(y_real.max(), y_predicho.max()))
        fig = go.Figure()
        fig.add_scatter(
            x=y_real.tolist(), y=y_predicho.tolist(),
            mode="markers",
            marker=dict(color=PALETTE[0], opacity=0.6, size=6),
            name="Predicciones",
        )
        fig.add_scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color=PALETTE[1], dash="dash", width=1.5),
            name="Ideal",
        )
        fig.update_layout(
            template="daad",
            xaxis_title="Real",
            yaxis_title="Predicho",
        )
        return fig

    def grafica_residuos(self, y_real: np.ndarray, y_predicho: np.ndarray) -> go.Figure:
        """Residuo (real − predicho) frente al valor predicho.

        En un buen modelo los residuos se reparten al azar alrededor de 0; un
        patrón (curva, embudo) delata que el modelo se está dejando estructura.
        """
        y_real = np.asarray(y_real, dtype=float)
        y_predicho = np.asarray(y_predicho, dtype=float)
        residuos = y_real - y_predicho
        fig = go.Figure()
        fig.add_scatter(
            x=y_predicho.tolist(), y=residuos.tolist(),
            mode="markers",
            marker=dict(color=PALETTE[0], opacity=0.6, size=6),
            name="Residuos",
        )
        fig.add_hline(y=0, line=dict(color=PALETTE[1], dash="dash", width=1.5))
        fig.update_layout(
            template="daad",
            xaxis_title="Predicho",
            yaxis_title="Residuo (real − predicho)",
        )
        return fig

    def grafica_confusion(self, matriz: np.ndarray, etiquetas: list | None = None) -> go.Figure:
        n = len(matriz)
        labels = etiquetas if etiquetas else [str(i) for i in range(n)]
        fig = go.Figure(
            go.Heatmap(
                z=matriz,
                x=labels,
                y=labels,
                colorscale=[[0, "#fafaf7"], [1, "#d2502a"]],
                text=matriz,
                texttemplate="%{text}",
                hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Conteo: %{z}<extra></extra>",
            )
        )
        fig.update_layout(
            template="daad",
            xaxis_title="Predicho",
            yaxis_title="Real",
        )
        return fig

    def grafica_roc(self, y_real: np.ndarray, proba: np.ndarray, clases) -> go.Figure:
        """Curva ROC (TPR vs FPR). Binario: una curva; multiclase: una por clase (uno-vs-resto).

        Más arriba/izquierda = mejor; la diagonal es el azar. La leyenda muestra el AUC.
        """
        y_real = np.asarray(y_real)
        clases = list(clases)
        fig = go.Figure()
        if len(clases) == 2:
            fpr, tpr, _ = roc_curve(y_real, proba[:, 1], pos_label=clases[1])
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc(fpr, tpr):.3f}",
                            line=dict(color=PALETTE[0], width=2))
        else:
            for i, c in enumerate(clases):
                fpr, tpr, _ = roc_curve((y_real == c).astype(int), proba[:, i])
                fig.add_scatter(x=fpr, y=tpr, mode="lines",
                                name=f"{c} (AUC = {auc(fpr, tpr):.3f})",
                                line=dict(color=PALETTE[i % len(PALETTE)], width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar",
                        line=dict(color="#9a9a9a", dash="dash", width=1.5))
        fig.update_layout(template="daad", xaxis_title="Tasa de falsos positivos",
                          yaxis_title="Tasa de verdaderos positivos")
        return fig

    def grafica_precision_recall(self, y_real: np.ndarray, proba: np.ndarray, clases) -> go.Figure:
        """Curva Precisión-Recall. Binario: una curva; multiclase: una por clase (uno-vs-resto).

        Útil sobre todo con clases desbalanceadas; la leyenda muestra la precisión media (AP).
        """
        y_real = np.asarray(y_real)
        clases = list(clases)
        fig = go.Figure()
        if len(clases) == 2:
            y_bin = (y_real == clases[1]).astype(int)
            prec, rec, _ = precision_recall_curve(y_bin, proba[:, 1])
            ap = average_precision_score(y_bin, proba[:, 1])
            fig.add_scatter(x=rec, y=prec, mode="lines", name=f"AP = {ap:.3f}",
                            line=dict(color=PALETTE[0], width=2))
        else:
            for i, c in enumerate(clases):
                y_bin = (y_real == c).astype(int)
                prec, rec, _ = precision_recall_curve(y_bin, proba[:, i])
                ap = average_precision_score(y_bin, proba[:, i])
                fig.add_scatter(x=rec, y=prec, mode="lines",
                                name=f"{c} (AP = {ap:.3f})",
                                line=dict(color=PALETTE[i % len(PALETTE)], width=2))
        fig.update_layout(template="daad", xaxis_title="Recall", yaxis_title="Precisión")
        return fig
