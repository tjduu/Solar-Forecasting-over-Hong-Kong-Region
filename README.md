# Solar-Forecasting-over-Hong-Kong-Region
This is the repo for spatial-temporal solar forecasting over Hong Kong region, including data download, data preprocessing, model training, and model prediction.

## Environment setup

Pick **one** workflow:

**Conda (recommended)**
```bash
mamba env create -f environment.yml   # or conda env create ...
mamba activate solar-forecasting-hk
```

**Pip/virtualenv**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation (high level)
1. Download PTREE/crops and CAMS datasets into `Data/`.
2. Use `src/data_preprocessing.py` helpers or the notebooks (`prepare_raw_data.ipynb`, `Prepare_Train_data.ipynb`) to:
   - match CAMS to PTREE timestamps,
   - build band/elevation tensors and CSI targets (`Data/output/*.npz`).

## Training

*CNNUNet/Twin (image-to-image)*
```python
from torch.utils.data import DataLoader
from src.dataset import BandsCamsDataset
from src.model_Twin import TwinFiLMUNetTiny
from src.train_val_bc import train_one_epoch, evaluate

ds = BandsCamsDataset(X, y, channel_min, channel_max, elev_ch_idx, land_ch_idx)
ldr = DataLoader(ds, batch_size=4, shuffle=True)
model = TwinFiLMUNetTiny().cuda()
# run train_one_epoch/evaluate loops as in the notebooks
```

## Evaluation & visualization
- Use `src/evaluate.py` for metrics (`evaluate_model`, `evaluate_daytime`) and sequence plots (`plot_sequence`).
- Use `src/plot.py` for full-image CSI/GHI visualization and merged day/night sequences.

## Notebooks
- `Train_Model.ipynb`, `Prepare_Train_data.ipynb`: end-to-end training flows for CNN models.
- `prepare_raw_data.ipynb`: data pairing and preprocessing.
- `test_location_consistency.ipynb`: location sanity checks.