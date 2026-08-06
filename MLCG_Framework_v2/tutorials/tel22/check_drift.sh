uv run \
  --with numpy \
  --with matplotlib \
  python check_energy_drift.py \
  energy.csv \
  --dt 0.001 \
  --strict
