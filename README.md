## Ocean Plastic Drift Prediction (Spatio-Temporal Attention)

This repository implements a spatio-temporal deep learning pipeline to forecast ocean plastic accumulation zones using multi-modal data (satellite, oceanographic, and historical drift). It includes a runnable synthetic demo so you can train, evaluate, visualize, and explore via a Streamlit dashboard.

### Quickstart (CPU)

1) Create a virtual environment and install deps:

```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# For CPU-only torch on Windows, install from the CPU wheel index if needed:
# pip install torch==2.2.2+cpu -i https://download.pytorch.org/whl/cpu
```

2) Run the end-to-end synthetic demo:

```bash
python -m src.train --config configs/config.yaml
python -m src.evaluate --config configs/config.yaml --checkpoint data/checkpoints/best.pt
python -m src.visualize --config configs/config.yaml --checkpoint data/checkpoints/best.pt
python -m src.run_trajectories --config configs/config.yaml
```

3) Launch the Streamlit dashboard:

```bash
streamlit run src/dashboard.py
```

### Docker (CPU)

Build and run the dashboard (serves at http://localhost:8501):

```bash
docker build -t plastic-drift:cpu .
docker run --rm -p 8501:8501 -v %cd%/data:/app/data plastic-drift:cpu
```

Train inside Docker (optional):

```bash
docker run --rm -v %cd%:/app plastic-drift:cpu \
  python -m src.train --config configs/config.yaml
```

### Project Structure

- `configs/config.yaml`: All paths and hyperparameters
- `src/data/`: Data loading and synthetic data generation
- `src/models/`: Spatio-temporal attention network
- `src/utils/`: Config, metrics, helpers
- `src/train.py`: Training script
- `src/evaluate.py`: Evaluation script and metrics
- `src/visualize.py`: Forecast heatmaps and animations
- `src/trajectory.py`: Simple particle advection for release-point trajectories
- `src/run_trajectories.py`: CLI to run trajectory simulation on synthetic velocity
- `src/dashboard.py`: Streamlit app (load checkpoint, visualize predictions, run trajectories)

### Notes on Real Data
This repo focuses on the modeling and pipeline. Integrate real data by extending `src/data/dataset.py` to read satellite imagery (e.g., GeoTIFF via Rasterio), oceanographic fields (NetCDF/xarray; HYCOM/OSCAR currents, ERA5 winds, SST/SSS), and historical trajectories (drifters/NGOs). Align temporally, resample to a common grid, and stack channels to match `data.channels.count`.

### License
MIT
