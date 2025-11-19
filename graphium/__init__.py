"""
Graphium: A clean EDA visualization toolkit
Provides:
- Static plots using Matplotlib/Seaborn
- Interactive plots using Plotly
"""

# STATIC PLOTS
from .static import (
    sp_cat_univariate,
    sp_num_univariate,
    sp_feature_target_analysis
)

# INTERACTIVE PLOTS
from .interactive import (
    ip_cat_univariate,
    ip_num_univariate,
    ip_cat_vs_target
)

__all__ = [
    # Static
    "sp_cat_univariate",
    "sp_num_univariate",
    "sp_feature_target_analysis",

    # Interactive
    "ip_cat_univariate",
    "ip_num_univariate",
    "ip_cat_vs_target",
]
