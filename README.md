# Robust Developed-Market Factor Allocation

This project studies whether long-only developed-market factor indices can be combined more robustly than simple allocation rules and the MSCI World benchmark. The evaluation is a historical walk-forward exercise, not a genuinely untouched out-of-sample test.

The factor set contains Value, Momentum, Quality, Minimum Volatility and MSCI World Equal Weighted. Equal Weight is used as an approximate size tilt, not as a pure size factor.

## Setup

In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

Raw MSCI files are not included. They must be obtained independently and used under the applicable terms. Place the required files in `data/raw` before running the first notebook. The expected names are listed in `data/raw/README.md`. See `DATA_NOTICE.md` before using or sharing data and generated outputs.

## Run order

Run the notebooks in numerical order:

1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/02_exploratory_analysis.ipynb`
3. `notebooks/03_static_allocations.ipynb`
4. `notebooks/04_rolling_optimization.ipynb`
5. `notebooks/05_out_of_sample_evaluation.ipynb`
6. `notebooks/06_robustness_analysis.ipynb`

The first notebook creates `monthly_index_levels.csv`, `monthly_returns_full.csv` and `monthly_returns_common.csv` in `data/processed`.

Notebooks 03 to 06 save tables, portfolio weights and figures in `results`. These files remain local and are not part of the public code repository.

Run the tests with:

```powershell
python -m unittest discover -s tests -v
```

## Results

The default specification and robustness settings are stored in `config/config.yaml`. Result tables and figures are generated locally and are not included in the public code-only repository.

Further details are in `docs/research_design.md`, `docs/methodology.md`, `docs/data_sources.md` and `docs/limitations.md`.
