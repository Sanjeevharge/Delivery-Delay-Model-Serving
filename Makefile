# Linux/macOS/CI convenience. On Windows use the PowerShell commands in README_U2.md
sample:   ; python make_sample_data.py --out data_sample
train:    ; python -m u2.train --data data --out artifacts
test:     ; python -m pytest -q tests
serve:    ; docker compose up --build
monitor:  ; python -m u2.monitor --scenario clean --out monitor_out && python -m u2.monitor --scenario drift --out monitor_out
loadtest: ; python -m u2.loadtest --rate 50 --seconds 30
