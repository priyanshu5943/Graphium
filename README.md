# Graphium

Graphium is a lightweight toolkit for static (Matplotlib/Seaborn) and interactive (Plotly) exploratory data analysis. It provides fast and clean visualization functions for understanding datasets.

## Features

- **Static EDA (Matplotlib/Seaborn)**
  - Categorical univariate plots
  - Numeric univariate plots
  - Feature–target relationship analysis

- **Interactive EDA (Plotly)**
  - Interactive categorical univariate analysis
  - Interactive numeric univariate analysis
  - Categorical feature vs target visualization

## Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/priyanshu5943/Graphium.git
```

## Usage

### Import Graphium
```python
import graphium as gh
```

### Static Visualizations
```python
# Categorical univariate
gh.sp_cat_univariate(df)

# Numeric univariate
gh.sp_num_univariate(df)

# Feature–target analysis
gh.sp_feature_target_analysis(df, target_col='loan_paid_back')
```

### Interactive Visualizations
```python
# Categorical univariate (interactive)
gh.ip_cat_univariate(df)

# Numeric univariate (interactive)
gh.ip_num_univariate(df)

# Categorical feature vs target (interactive)
gh.ip_cat_vs_target(df, target_col='loan_paid_back')
```