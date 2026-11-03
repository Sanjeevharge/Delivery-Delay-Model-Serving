# Delivery-delay prediction (Olist)

**Problem:** when an order is placed, predict whether it will arrive **after the date promised to the customer**, so operations can step in early (expedite, switch carrier, nudge the seller, warn the customer).

## Files
| File | What it is |
|---|---|
| `delivery_delay.py` | The full pipeline: data quality → target → leakage audit → EDA → features → time split → baseline → models → feature selection → calibration → cost curve → error analysis → SHAP |
| `make_sample_data.py` | Makes **synthetic** data in the exact Olist format, so you can test the code without the real download |
| `data/` | Put the real Olist CSVs here (you create this folder) |
| `outputs/` | Created by the script: `report.md`, plots, CSVs, `test_predictions.csv` |

## Run it (PowerShell)
```powershell
cd C:\Users\Sanjeev\Downloads\udaan\delivery_delay_prediction

# 1) Smoke test on synthetic data (about 15 seconds)
C:\Users\Sanjeev\anaconda3\python.exe make_sample_data.py --out data_sample
C:\Users\Sanjeev\anaconda3\python.exe delivery_delay.py --data data_sample --out outputs_sample

# 2) Real data: download the zip from
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
#    and extract all CSVs into .\data   then:
C:\Users\Sanjeev\anaconda3\python.exe delivery_delay.py --data data --out outputs

# Optional: change the business costs (default: false alarm = 1, missed delay = 5)
C:\Users\Sanjeev\anaconda3\python.exe delivery_delay.py --data data --out outputs --cost-false-alarm 1 --cost-missed-delay 10
```
Then open `outputs\report.md` (in VS Code: right-click → *Open Preview*).

## Key decisions
- **Target:** `late = delivered calendar day > order_estimated_delivery_date`.
- **Cancelled / unavailable / undelivered orders are excluded**: they never got a delivery, so "late" is undefined. Their counts are listed in the report.
- **Last 45 days dropped:** in those weeks only the fast orders have a delivery date yet, so keeping them would make the late rate look too low.
- **Leakage:** delivery and carrier dates, order status and reviews are never used as features.
- **Seller late rate** only counts orders that were delivered *before* the current order was bought.
- **Split by purchase date:** 70 / 15 / 15. The model is chosen on validation; test is only used for reporting.
