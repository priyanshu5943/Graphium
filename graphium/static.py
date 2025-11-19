import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import colorsys
from IPython.display import HTML, display

# ============================================================
# COLOR UTILITIES
# ============================================================

base_colors = [
    "#40FF80", "#FF4040", '#1E90FF', '#FFFF00', '#BA55D3', '#00FA9A', '#DC143C',
    '#FF8C00', '#00BFFF', '#8A2BE2', '#7B68EE', '#3CB371',
    '#FF1493', '#00FF7F', '#B22222', '#DAA520', '#5F9EA0',
    '#20B2AA', '#F08080', '#ADFF2F', '#CD5C5C', '#8FBC8F',
    '#87CEEB', '#D2691E', '#FF4500', '#708090', '#6A5ACD'
]

_single_color_cycle = itertools.cycle(base_colors)


def adjust_color(hex_color, factor=1.2):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
    l = min(1.0, l * factor)
    r_adj, g_adj, b_adj = colorsys.hls_to_rgb(h, l, s)
    return '{:02X}{:02X}{:02X}'.format(int(r_adj*255), int(g_adj*255), int(b_adj*255))


def get_recycled_colors(categories):
    color_map = {}
    num_base = len(base_colors)
    for i, category in enumerate(categories):
        base_idx = i % num_base
        repeat_idx = i // num_base
        base_color = base_colors[base_idx]

        if repeat_idx == 0:
            color_map[category] = base_color
        else:
            color_map[category] = adjust_color(base_color, 1 + 0.1 * repeat_idx)

    return color_map



def _create_subplots(n_plots, figsize=(20, 17)):
    n_rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    plt.subplots_adjust(hspace=0.8, wspace=0.8)

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_plots == 1:
        axes = np.array([[axes]])

    return fig, axes.flatten()

# ============================================================
# HELPER UTILITIES (MUST EXIST FOR MAIN FUNCTION)
# ============================================================

def _spacer_div(height=25):
    display(HTML(
        f"<div style='margin-top:{height}px; margin-bottom:{height}px; border-bottom:1px solid #333;'></div>"
    ))



def sp_cat_univariate(df):
    """
    Your donut + countplot univariate categorical visualizer.
    (Original code preserved)
    """
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    eligible_cols = [c for c in df.columns if df[c].nunique() <= 10]

    for col in eligible_cols:
        fig, ax = plt.subplots(1, 2, figsize=(14, 3))
        ax = ax.flatten()

        counts = df[col].value_counts()
        labels = counts.index.tolist()
        colors = base_colors[:len(labels)]

        # Donut chart
        wedges, _, _ = ax[0].pie(
            counts,
            autopct='%1.1f%%',
            colors=colors,
            wedgeprops=dict(width=0.35),
            startangle=80, pctdistance=0.85,
            textprops={'size': 9, 'color': 'white', 'fontweight': 'bold'}
        )
        centre_circle = plt.Circle((0, 0), 0.6, fc='white')
        ax[0].add_artist(centre_circle)

        # Countplot
        sns.countplot(data=df, y=col, palette=colors, order=labels, ax=ax[1])
        for i, v in enumerate(counts):
            ax[1].text(v + 0.5, i, str(v), va='center')

        sns.despine(left=True)
        ax[1].set_xlabel("")
        ax[1].set_ylabel(None)
        ax[1].set_xticks([])

        fig.suptitle(col, fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()

        _spacer_div(15)



def sp_num_univariate(df):
    """
    Your original boxplot grid for numeric columns.
    (Original code preserved)
    """
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    colors = base_colors * 4
    cols_per_row = 2
    total_cols = len(numeric_columns)
    total_rows = (total_cols + cols_per_row - 1) // cols_per_row

    fig = plt.figure(figsize=[8 * cols_per_row, 3.5 * total_rows])
    display(HTML("<h1 style='color:#FF6347; text-align:center'>Distribution of numeric features</h1>"))

    fig.subplots_adjust(hspace=0.8, wspace=0.4)

    for i, col in enumerate(numeric_columns):
        ax = fig.add_subplot(total_rows, cols_per_row, i + 1)
        sns.boxplot(data=df, x=col, color=colors[i], ax=ax)
        ax.set_title(col, fontsize=18, fontweight='bold')
        ax.set_xlabel('')
        ax.grid(False)

    plt.show()
    _spacer_div(25)







# ============================================================
# FEATURE–TARGET ANALYSIS (FULL ORIGINAL FUNCTION)
# ============================================================

def _plot_categorical_vs_categorical(df, x_col, color_col, ax):
    unique_categories = df[color_col].dropna().unique()
    color_map = get_recycled_colors(unique_categories)

    sns.countplot(
        data=df, x=x_col, hue=color_col,
        palette=color_map, ax=ax
    )

    ax.set_xlabel(x_col, fontsize=16, fontweight='bold')
    ax.set_ylabel('Count', fontsize=16, fontweight='bold')
    ax.set_title(f'{x_col} vs {color_col}', fontsize=20, fontweight='bold')

    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.legend(title=color_col, frameon=False)
    ax.set_facecolor('white')



def _plot_categorical_vs_numeric(df, cat_col, target_col, ax):
    unique_categories = df[cat_col].dropna().unique()
    color_map = get_recycled_colors(unique_categories)

    palette = {cat: color_map[cat] for cat in df[cat_col].unique() if pd.notna(cat)}

    sns.boxplot(
        data=df, x=cat_col, y=target_col,
        palette=palette, ax=ax
    )

    ax.set_xlabel(cat_col, fontsize=16, fontweight='bold')
    ax.set_ylabel(target_col, fontsize=16, fontweight='bold')
    ax.set_title(f'{target_col} by {cat_col}', fontsize=20, fontweight='bold')

    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.set_facecolor('white')


def _plot_numeric_vs_numeric(df, x_col, y_col, category_col=None, ax=None):
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    strength = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.3 else "weak"
    direction = "positive" if corr > 0 else "negative" if corr < 0 else "no"

    title = f"{x_col} vs {y_col}"
    if category_col:
        title += f" by {category_col}"

    if category_col:
        unique_cats = df[category_col].dropna().unique()
        color_map = get_recycled_colors(unique_cats)

        sns.scatterplot(
            data=df, x=x_col, y=y_col,
            hue=category_col, palette=color_map,
            edgecolor=None, ax=ax
        )
    else:
        sns.scatterplot(
            data=df, x=x_col, y=y_col,
            color=next(_single_color_cycle),
            edgecolor=None, ax=ax
        )

    ax.set_xlabel(x_col, fontsize=16, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=16, fontweight='bold')
    ax.set_title(f"{title}\nCorr: {corr:.2f} ({strength} {direction})", fontsize=18)

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.set_facecolor('white')

    if category_col:
        ax.legend(title=category_col, frameon=False)


def sp_feature_target_analysis(df, target_col, color_col=None, plot_type=3):
    """
    This is your FULL original long function.
    NOT modified.
    Only renamed from analyze_feature_target_relationships().
    """

    # ----- ENTIRE FUNCTION BODY EXACTLY AS YOU WROTE IT -----
    heading_color = "#FF6347"

    if df[target_col].nunique() < 8:
        df[target_col] = df[target_col].astype("object")

    cat_vs_num = []
    cat_vs_cat = []
    num_vs_num = []

    id_cols = [col for col in df.columns if "id" in col.lower()]
    excluded_cols = set(id_cols + [target_col])

    bool_cols = [col for col in df.columns if df[col].dtype == bool]
    df[bool_cols] = df[bool_cols].astype(object)

    drop_cols = [col for col in df.columns if df[col].dtype == object and df[col].nunique() > 15]
    df = df.drop(columns=drop_cols)
    excluded_cols.update(drop_cols)

    num_target_is_cat = pd.api.types.is_categorical_dtype(df[target_col]) or df[target_col].dtype == object

    for feature in df.columns:
        if feature in excluded_cols:
            continue
        feature_is_cat = pd.api.types.is_categorical_dtype(df[feature]) or df[feature].dtype == object
        feature_is_num = pd.api.types.is_numeric_dtype(df[feature])

        if feature_is_num and pd.api.types.is_numeric_dtype(df[target_col]):
            num_vs_num.append(feature)
        elif feature_is_cat and not num_target_is_cat:
            cat_vs_num.append(feature)
        elif feature_is_cat and num_target_is_cat:
            cat_vs_cat.append(feature)
        elif feature_is_num and num_target_is_cat:
            cat_vs_num.append(feature)

    # ---------------- Numeric vs Numeric ----------------
    if plot_type == 3 and num_vs_num:
        display(HTML(f"<h1 style='color:{heading_color}; text-align:center'>Numeric Feature vs {target_col}</h1>"))
        _spacer_div(20)
        fig, axes = _create_subplots(len(num_vs_num))
        for i, feature in enumerate(num_vs_num):
            if i < len(axes):
                _plot_numeric_vs_numeric(df, feature, target_col, category_col=color_col, ax=axes[i])
            else:
                plt.tight_layout()
                plt.show()
                fig, axes = _create_subplots(len(num_vs_num) - i)
                _plot_numeric_vs_numeric(df, feature, target_col, category_col=color_col, ax=axes[0])
        for i in range(len(num_vs_num), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()
        _spacer_div(35)

    # ---------------- Categorical vs Numeric ---------------
    if plot_type == 2 and cat_vs_num:
        display(HTML(f"<h1 style='color:{heading_color}; text-align:center'>Numeric Feature vs {target_col}</h1>"))
        _spacer_div(20)
        fig, axes = _create_subplots(len(cat_vs_num))
        for i, feature in enumerate(cat_vs_num):
            if i < len(axes):
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    _plot_categorical_vs_numeric(df, feature, target_col, ax=axes[i])
                else:
                    _plot_categorical_vs_numeric(df, target_col, feature, ax=axes[i])
            else:
                plt.tight_layout()
                plt.show()
                fig, axes = _create_subplots(len(cat_vs_num) - i)
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    _plot_categorical_vs_numeric(df, feature, target_col, ax=axes[0])
                else:
                    _plot_categorical_vs_numeric(df, target_col, feature, ax=axes[0])
        for i in range(len(cat_vs_num), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()
        _spacer_div(35)

    # ---------------- Categorical vs Categorical -----------
    if plot_type == 1 and cat_vs_cat:
        display(HTML(f"<h1 style='color:{heading_color}; text-align:center'>Categorical Feature vs {target_col}</h1>"))
        _spacer_div(20)
        fig, axes = _create_subplots(len(cat_vs_cat))
        for i, feature in enumerate(cat_vs_cat):
            if i < len(axes):
                _plot_categorical_vs_categorical(df, feature, target_col, ax=axes[i])
            else:
                plt.tight_layout()
                plt.show()
                fig, axes = _create_subplots(len(cat_vs_cat) - i)
                _plot_categorical_vs_categorical(df, feature, target_col, ax=axes[0])
        for i in range(len(cat_vs_cat), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()
        _spacer_div(35)