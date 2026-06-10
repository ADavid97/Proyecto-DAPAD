import plotly.graph_objects as go
import plotly.io as pio

_ACCENT  = "#d2502a"
_PAPER   = "#fafaf7"
_PAPER2  = "#f3f0e8"
_INK     = "#1a1a1a"
_PENCIL  = "#6b6b6b"
_GHOST   = "#d8d4cb"

PALETTE = ["#d2502a", "#2a6fbb", "#2a8a4a", "#8a3aa8", "#c98a2a", "#6b6b6b"]

_t = go.layout.Template()
_t.layout = go.Layout(
    font=dict(family="'IBM Plex Sans', Helvetica, sans-serif", size=12, color=_INK),
    paper_bgcolor=_PAPER,
    plot_bgcolor=_PAPER,
    colorway=PALETTE,
    title=dict(font=dict(size=14, color=_INK), x=0, xanchor="left", pad=dict(b=8)),
    xaxis=dict(
        gridcolor=_GHOST, linecolor=_GHOST, tickcolor=_GHOST, zeroline=False,
        tickfont=dict(family="'IBM Plex Mono', monospace", size=10, color=_PENCIL),
        title=dict(font=dict(size=11, color=_PENCIL)),
    ),
    yaxis=dict(
        gridcolor=_GHOST, linecolor=_GHOST, tickcolor=_GHOST, zeroline=False,
        tickfont=dict(family="'IBM Plex Mono', monospace", size=10, color=_PENCIL),
        title=dict(font=dict(size=11, color=_PENCIL)),
    ),
    legend=dict(
        bgcolor=_PAPER2, bordercolor=_GHOST, borderwidth=1,
        font=dict(size=11, color=_INK),
    ),
    hoverlabel=dict(
        bgcolor=_PAPER, bordercolor=_GHOST,
        font=dict(family="'IBM Plex Mono', monospace", size=11, color=_INK),
    ),
    margin=dict(l=44, r=20, t=40, b=44),
)

pio.templates["daad"] = _t
pio.templates.default = "daad"
