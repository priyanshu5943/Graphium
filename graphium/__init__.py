"""
Graphium: A clean EDA visualization toolkit

Modules:
- Static plots (Matplotlib / Seaborn)
- Interactive plots (Plotly)
- Multivariate plots (static + interactive)
"""

# ---------------------------------------------------------
# STATIC PLOTS
# ---------------------------------------------------------
from .static import (
    sp_cat_univariate,
    sp_num_univariate,
    sp_feature_target_analysis
)

# ---------------------------------------------------------
# INTERACTIVE PLOTS
# ---------------------------------------------------------
from .interactive import (
    ip_cat_univariate,
    ip_num_univariate,
    ip_cat_vs_target
)

# ---------------------------------------------------------
# MULTIVARIATE PLOTS
# ---------------------------------------------------------
from .multivariate import (
    mv_num_cat_vs_target_grid,
    mv_corr_heatmap
)

# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------
__all__ = [
    # Static
    "sp_cat_univariate",
    "sp_num_univariate",
    "sp_feature_target_analysis",

    # Interactive
    "ip_cat_univariate",
    "ip_num_univariate",
    "ip_cat_vs_target",

    # Multivariate
    "mv_num_cat_vs_target_grid",
    "mv_corr_heatmap",
]
