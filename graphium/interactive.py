import numpy as np
import pandas as pd
from scipy.stats import entropy
from IPython.display import display, HTML

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================
# CATEGORICAL INTERACTIVE (UNIVARIATE)
# ============================================================

def ip_cat_univariate(df):
    """
    Fully interactive univariate visualization for categorical features.
    Produces:
        • Summary statistics
        • Top & bottom category tables
        • Count bar chart
        • Percentage donut chart
    Layout is fixed to ensure correct vertical stacking in both
    live notebooks and saved notebooks.
    """

    feature_print_count = {}

    # ----------------------------------------
    # Helper: detect categorical
    # ----------------------------------------
    def is_categorical(col):
        return (
            (df[col].dtype == 'object' and df[col].nunique() <= 15) or
            df[col].dtype.name == 'category' or
            (np.issubdtype(df[col].dtype, np.number) and df[col].nunique() <= 15)
        )

    # ----------------------------------------
    # Helper: summary block
    # ----------------------------------------
    def summarize_categorical(series, feature):
        value_counts = series.value_counts(dropna=False)
        percentage = (value_counts / value_counts.sum()) * 100

        num_unique = value_counts.shape[0]
        total_rows = len(series)
        missing_count = series.isna().sum()
        missing_pct = (missing_count / total_rows) * 100

        most_cat = value_counts.idxmax()
        least_cat = value_counts.idxmin()

        imbalance_ratio = value_counts.max() / max(value_counts.min(), 1)

        ent_val = entropy(value_counts)
        p = value_counts / value_counts.sum()
        gini_index = 1 - (p ** 2).sum()

        mode_ratio = value_counts.max() / total_rows

        feature_print_count[feature] = feature_print_count.get(feature, 0) + 1
        font_size = 24 + 4 * feature_print_count[feature]

        # ----------------------------------------
        # MAIN HEADER
        # ----------------------------------------
        display(HTML("<div style='clear:both; width:100%; height:25px;'></div>"))
        display(HTML(
            f"<h2 style='text-align:center; font-size:{font_size}px; color:green;'>"
            f"<b>{feature}</b></h2>"
        ))

        # ----------------------------------------
        # Summary text
        # ----------------------------------------
        print(f"{'Total Rows':<30}: {total_rows}")
        print(f"{'Unique Categories':<30}: {num_unique}")
        print(f"{'Missing Values':<30}: {missing_count} ({missing_pct:.2f}%)")
        print(f"{'Most Frequent Category':<30}: {most_cat} "
              f"({value_counts.max()} | {percentage[most_cat]:.2f}%)")
        print(f"{'Least Frequent Category':<30}: {least_cat} "
              f"({value_counts.min()} | {percentage[least_cat]:.2f}%)")
        print(f"{'Imbalance Ratio (max/min)':<30}: {imbalance_ratio:.2f}")
        print(f"{'Entropy':<30}: {ent_val:.2f}")
        print(f"{'Gini Index':<30}: {gini_index:.2f}")
        print(f"{'Mode Frequency Ratio':<30}: {mode_ratio:.2f}")

        # ----------------------------------------
        # Top & bottom categories
        # ----------------------------------------
        top_n = 1 if num_unique <= 5 else (3 if num_unique <= 20 else 5)

        top_html = (
            "<ul>" +
            "".join([
                f"<li>{i+1}. {k} - {v} ({percentage[k]:.2f}%)</li>"
                for i, (k, v) in enumerate(value_counts.head(top_n).items())
            ]) +
            "</ul>"
        )

        bottom_html = (
            "<ul>" +
            "".join([
                f"<li>{i+1}. {k} - {v} ({percentage[k]:.2f}%)</li>"
                for i, (k, v) in enumerate(value_counts.tail(top_n).items())
            ]) +
            "</ul>"
        )

        table_html = f"""
        <table style='width:100%; margin-top:15px; table-layout:fixed; font-size:14px;'>
            <tr>
                <th style='text-align:center; color:#00CED1;'>Top Categories</th>
                <th style='text-align:center; color:#FF69B4;'>Bottom Categories</th>
            </tr>
            <tr>
                <td style='vertical-align:top; padding:10px;'>{top_html}</td>
                <td style='vertical-align:top; padding:10px;'>{bottom_html}</td>
            </tr>
        </table>
        """

        display(HTML(table_html))

        # Force vertical stacking before plots
        # Small spacing before plots
        display(HTML("<div style='clear:both; width:100%; height:10px;'></div>"))

        return value_counts, percentage

    # ----------------------------------------
    # Helper: generate interactive plot
    # ----------------------------------------
    def plot_categorical_distribution(value_counts, percentage, feature):

        rotate_labels = -45 if len(value_counts) > 8 else 0

        colors = [
            "#40FF80", "#FF4040", '#FFD700', '#40E0D0', '#FF69B4', '#7FFFD4',
            '#FFA500', '#00FA9A', '#FF4500', '#4682B4', '#DA70D6',
            '#FFB6C1', '#FF1493', '#FF8C00', '#98FB98', '#9370DB',
            '#32CD32', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'
        ]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f"{feature} - Count", f"{feature} - % Share"),
            specs=[[{"type": "bar"}, {"type": "domain"}]],
        )

        # Bar chart
        fig.add_trace(
            go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                text=value_counts.values,
                textposition="outside",
                marker_color=colors[:len(value_counts)],
            ),
            row=1, col=1
        )

        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=value_counts.index.astype(str),
                values=percentage,
                hole=0.45,
                marker_colors=colors[:len(value_counts)],
            ),
            row=1, col=2
        )

        fig.update_layout(
            title=f"Distribution of {feature}",
            title_font=dict(size=18),
            showlegend=False,
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            height=300,
            width=500,
            margin=dict(t=40, l=20, r=20, b=20),
        )

        fig.update_xaxes(tickangle=rotate_labels)

        fig.show()

        # Minimal vertical spacing (tight layout)
        display(HTML("<div style='clear:both; width:100%; height:15px;'></div>"))
        display(HTML("<hr style='border:0; border-top:1px solid #aaa; margin:15px 0;'>"))



    # ----------------------------------------
    # MAIN LOOP
    # ----------------------------------------
    for feature in df.columns:
        if feature.lower() == "id":
            continue

        if is_categorical(feature):
            series = df[feature]
            value_counts, percentage = summarize_categorical(series, feature)
            plot_categorical_distribution(value_counts, percentage, feature)







# ============================================================
# INTERACTIVE NUMERIC UNIVARIATE
# ============================================================

def ip_num_univariate(df):
    """
    Interactive univariate visualization for numeric features.
    Produces:
        - Summary statistics (skew, kurtosis, IQR, CV, outliers, etc.)
        - Histogram
        - Violin Plot
        - Automatic skipping of ID-like columns
        - Pretty HTML formatting
    """

    # --------------------------
    # Identify numeric columns
    # --------------------------
    id_like_keywords = {'id', 'index', 'serial'}
    exclude_cols = {col for col in df.columns if col.lower() in id_like_keywords}
    numeric_cols = [col for col in df.select_dtypes(include='number').columns if col not in exclude_cols]

    feature_print_count = {}

    # --------------------------
    # Helper: Plot histogram + violin
    # --------------------------
    def create_combined_plot(data, feature, color, width=670, height=310):
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Histogram', 'Violin Plot')
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=data[feature],
                marker=dict(color=color),
                name='Histogram'
            ),
            row=1, col=1
        )

        # Violin Plot
        fig.add_trace(
            go.Violin(
                y=data[feature],
                box_visible=True,
                meanline_visible=True,
                line_color=color,
                name='Violin'
            ),
            row=1, col=2
        )

        # Section Labels
        annotations = [
            dict(text=f"<b>histogram</b>", x=0.22, y=1.1, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=16, color=color)),
            dict(text=f"<b>violin_plot</b>", x=0.78, y=1.1, xref='paper', yref='paper',
                 showarrow=False, font=dict(size=16, color=color))
        ]

        fig.update_layout(
            title_text=f"Distribution of {feature}",
            title_font=dict(size=20, family='Arial'),
            showlegend=False,
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            width=width,
            height=height,
            annotations=annotations,
            margin=dict(t=50, b=30, l=30, r=30)
        )

        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=False)

        fig.show()

    # --------------------------
    # Helper: Summary computation
    # --------------------------
    def numeric_feature_summary(df, feature):
        nonlocal feature_print_count

        series = df[feature]
        clean_series = series.dropna()

        total_rows = len(series)
        missing_count = series.isna().sum()
        missing_pct = (missing_count / total_rows) * 100

        min_val = clean_series.min()
        max_val = clean_series.max()
        mean_val = clean_series.mean()
        median_val = clean_series.median()
        std_val = clean_series.std()

        skew_val = skew(clean_series)
        kurt_val = kurtosis(clean_series)

        mode_val = clean_series.mode().iloc[0] if not clean_series.mode().empty else "N/A"

        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)].count()

        cv = std_val / mean_val if mean_val != 0 else float("nan")

        # Dynamic heading size
        feature_print_count[feature] = feature_print_count.get(feature, 0) + 1
        font_size = 24 + 4 * feature_print_count[feature]

        display(HTML(f"<div style='margin-bottom:40px;'>"))
        display(HTML(
            f"<h2 style='text-align:center; font-size:{font_size}px; color:green;'>"
            f"<b>{feature}</b></h2>"
        ))

        # Summary Table Data
        summary_data = [
            ("Total Rows", f"{total_rows}"),
            ("Missing Values", f"{missing_count} ({missing_pct:.2f}%)"),
            ("Mean", f"{mean_val:.2f}"),
            ("Median", f"{median_val:.2f}"),
            ("Standard Deviation", f"{std_val:.2f}"),
            ("Min", f"{min_val:.2f}"),
            ("Max", f"{max_val:.2f}"),
            ("Skewness", f"{skew_val:.2f}"),
            ("Kurtosis", f"{kurt_val:.2f}"),
            ("Mode", f"{mode_val}"),
            ("Q1 (25%)", f"{q1:.2f}"),
            ("Q3 (75%)", f"{q3:.2f}"),
            ("IQR", f"{iqr:.2f}"),
            ("Lower Bound (1.5*IQR)", f"{lower_bound:.2f}"),
            ("Upper Bound (1.5*IQR)", f"{upper_bound:.2f}"),
            ("Outlier Count (1.5*IQR)", f"{outlier_count}"),
            ("Coefficient of Variation", f"{cv:.2f}"),
        ]

        half = len(summary_data) // 2
        col1 = summary_data[:half]
        col2 = summary_data[half:]

        col1_html = "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in col1])
        col2_html = "".join([f"<li><b>{k}:</b> {v}</li>" for k, v in col2])

        table_html = f"""
        <table style='width:100%; font-size:14px; table-layout:fixed; margin-top:10px;'>
          <tr>
            <th style='text-align:center; width:50%; color:#00CED1;'>Summary</th>
            <th style='text-align:center; width:50%; color:#FF69B4;'>Details</th>
          </tr>
          <tr>
            <td style='vertical-align:top; padding: 10px;'><ul>{col1_html}</ul></td>
            <td style='vertical-align:top; padding: 10px;'><ul>{col2_html}</ul></td>
          </tr>
        </table>
        """
        display(HTML(table_html))

    # --------------------------
    # Color Palette
    # --------------------------
    color_palette = [
        '#FFD700', '#FFA500', '#00FA9A', '#FFB6C1', '#FF1493',
        'red', '#00CED1', '#1E90FF', '#FFFF00', '#7CFC00'
    ]

    # --------------------------
    # MAIN LOOP
    # --------------------------
    for i, feature in enumerate(numeric_cols):
        numeric_feature_summary(df, feature)
        create_combined_plot(
            df, feature,
            color_palette[i % len(color_palette)]
        )

        print("\n\n")



import plotly.express as px
from IPython.display import display, HTML
import colorsys

# ============================================================
# COLOR UTILITIES
# ============================================================

_base_colors = [
    '#FFD700', '#FF6347', '#40E0D0', '#FF69B4', '#4682B4', 'red',
    '#7CFC00', '#98FB98', '#9370DB', '#32CD32', '#00CED1',
    '#1E90FF', '#FFFF00', '#7CFC00'
]

def _adjust_color(hex_color, factor=1.2):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    l = min(1.0, l * factor)
    r_adj, g_adj, b_adj = colorsys.hls_to_rgb(h, l, s)
    return '#{0:02X}{1:02X}{2:02X}'.format(int(r_adj * 255), int(g_adj * 255), int(b_adj * 255))

def _get_recycled_colors(categories):
    color_map = {}
    num_base = len(_base_colors)
    for i, category in enumerate(categories):
        base_idx = i % num_base
        repeat_idx = i // num_base
        base_color = _base_colors[base_idx]
        if repeat_idx == 0:
            color_map[category] = base_color
        else:
            color_map[category] = _adjust_color(base_color, 1 + 0.1 * repeat_idx)
    return color_map

# ============================================================
# SINGLE PLOT FUNCTION
# ============================================================

def _plot_categorical_vs_target(df, x_col, target_col):
    unique_categories = df[target_col].dropna().unique()
    color_map = _get_recycled_colors(unique_categories)

    fig = px.histogram(
        df,
        x=x_col,
        color=target_col,
        barmode='group',
        color_discrete_map=color_map
    )

    fig.update_layout(
        title='',
        xaxis_title=x_col,
        yaxis_title='Count',
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='white', size=15),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='white', showline=False),
        yaxis=dict(showgrid=True, zeroline=True, zerolinecolor='white', showline=False),
        legend_title_text=target_col,
        legend_font=dict(color='white', size=12),
        width=500,
        height=300,
    )

    fig.show()

    # 🔥 Prevent horizontal rendering in saved notebooks
    display(HTML("<div style='clear:both; width:100%; height:15px;'></div>"))
    display(HTML("<hr style='border:0; border-top:1px solid #666; margin:15px 0;'>"))

# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def ip_cat_vs_target(df, target_col):
    """
    Interactive categorical feature vs target distribution plots.
    """

    # Detect categorical features except target
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_cols = [col for col in cat_cols if col != target_col]

    if not cat_cols:
        display(HTML("<h3 style='color:red;'>No categorical columns found for plotting.</h3>"))
        return

    # Header
    display(HTML(
        f"<h1 style='text-align:center; font-size:32px; color:green; margin-bottom:20px;'>"
        f"<b>Categorical Features vs Target Distribution</b></h1>"
    ))

    # Loop through categorical columns
    for feature in cat_cols:

        # Title for each feature
        display(HTML(f"<h2 style='text-align:center; font-size:22px; color:#FF69B4;'>"
                     f"<b>{target_col} by {feature}</b></h2>"))

        # Plot
        _plot_categorical_vs_target(df, x_col=feature, target_col=target_col)

        # 🔥 Minimal gap before next feature
        display(HTML("<div style='clear:both; width:100%; height:10px;'></div>"))

