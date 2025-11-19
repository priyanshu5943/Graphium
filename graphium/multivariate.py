import itertools
import colorsys
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML


# ---------------------------------------------------------
# Color Utilities
# ---------------------------------------------------------

def mv_adjust_color(hex_color, factor=1.2):
    """Brighten or darken a hex color by a given factor."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = min(1.0, l * factor)
    r_adj, g_adj, b_adj = colorsys.hls_to_rgb(h, l, s)
    return '#{0:02X}{1:02X}{2:02X}'.format(int(r_adj * 255),
                                          int(g_adj * 255),
                                          int(b_adj * 255))


def mv_get_recycled_colors(categories, base_colors):
    """Assign unique or adjusted colors to categories."""
    color_map = {}
    num_base = len(base_colors)

    for i, cat in enumerate(categories):
        base_idx = i % num_base
        repeat_idx = i // num_base
        base_color = base_colors[base_idx]

        if repeat_idx == 0:
            color_map[cat] = base_color
        else:
            color_map[cat] = mv_adjust_color(base_color, 1 + 0.1 * repeat_idx)

    return color_map


# ---------------------------------------------------------
# Main Multivariate Grid Plot
# ---------------------------------------------------------

def mv_num_cat_vs_target_grid(
    df,
    num_cols,
    cat_cols,
    target_col,
    plot_type='violin',
    n_cols=2
):
    """
    Grid of plots showing Numeric × Categorical combinations with
    target-based hue.

    plot_type: "violin", "box", or "swarm"
    """

    # Base color palette
    base_colors = [
        "#40FF80", "#FF4040", '#BA55D3', '#00FA9A', '#DC143C',
        '#FF8C00', '#00BFFF', '#8A2BE2', '#7B68EE', '#3CB371',
        '#FF1493', '#00FF7F', '#B22222', '#DAA520', '#5F9EA0',
        '#20B2AA', '#F08080', '#ADFF2F', '#CD5C5C', '#8FBC8F',
        '#87CEEB', '#D2691E', '#FF4500', '#708090', '#6A5ACD'
    ]

    heading_color = "#FF6347"

    # Build hue palette
    hue_values = df[target_col].dropna().unique()
    color_map = mv_get_recycled_colors(hue_values, base_colors)
    palette = {k: color_map[k] for k in hue_values}

    # Build grid combinations
    combos = list(itertools.product(num_cols, cat_cols))
    n_total = len(combos)
    n_rows = (n_total + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 9, n_rows * 6))
    axes = axes.flatten()

    # Plot all combinations
    for idx, (num_col, cat_col) in enumerate(combos):
        ax = axes[idx]

        try:
            if plot_type == 'violin':
                sns.violinplot(
                    data=df, x=cat_col, y=num_col,
                    hue=target_col, palette=palette,
                    split=True, ax=ax
                )

            elif plot_type == 'box':
                sns.boxplot(
                    data=df, x=cat_col, y=num_col,
                    hue=target_col, palette=palette, ax=ax
                )

            elif plot_type == 'swarm':
                sns.swarmplot(
                    data=df, x=cat_col, y=num_col,
                    hue=target_col, dodge=True,
                    palette=palette, ax=ax
                )

            else:
                raise ValueError("plot_type must be one of: 'violin', 'box', 'swarm'")

            ax.set_title(f"{num_col} by {cat_col}", fontsize=20, fontweight='bold')
            ax.set_xlabel(cat_col, fontsize=16, fontweight='bold')
            ax.set_ylabel(num_col, fontsize=16, fontweight='bold')
            ax.tick_params(axis='both', labelsize=14, width=1.5)
            ax.tick_params(axis='x', rotation=45)

            for tick in ax.get_xticklabels():
                tick.set_fontweight('bold')
            for tick in ax.get_yticklabels():
                tick.set_fontweight('bold')

        except Exception as e:
            ax.set_visible(False)
            print(f"Skipping {num_col} × {cat_col} due to error: {e}")

    # Hide remaining axes
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    # Title using HTML
    display(HTML(f"<h1 style='color:{heading_color}; text-align:center'>Multivariate Analysis</h1>"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.55, hspace=0.8)
    plt.show()







# -----------------------------------------------------------------------
# Correlation Heatmap (Interactive, Plotly)
# -----------------------------------------------------------------------

import plotly.graph_objects as go
import pandas as pd

def mv_corr_heatmap(df, title="Correlation Heatmap of Numeric Features"):
    """
    Interactive correlation heatmap for numeric features using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    title : str
        Title for the heatmap.
    """

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        raise ValueError("No numeric columns found for correlation heatmap.")

    # Compute correlation matrix
    correlation_matrix = numeric_df.corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        text=correlation_matrix.values,
        texttemplate="%{text}",
        colorscale="Tealrose",
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=title,
        height=700,
        width=900,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
    )

    fig.show()
