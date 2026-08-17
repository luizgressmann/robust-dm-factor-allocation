# Results

Generated results are not committed. Run the synthetic demo to create these files in `results/demo/`:

| File | Contents |
| --- | --- |
| `summary.csv` | Method-level statistics |
| `wealth.png` | Walk-forward wealth paths |
| `cap_sensitivity.png` | OAS-GMV cap check |
| `robustness.csv` | Full synthetic sensitivity grid |

Recreate the set with:

```powershell
python -m robust_dm_factor_allocation.demo --output-dir results/demo
```

The demo uses fixed-seed synthetic returns. The notebooks create separate MSCI-based results. Git ignores both sets of files.
