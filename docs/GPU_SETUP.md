## GPU Setup (PyTorch + PyG)

Due conflicts with system `/anaconda` on some cloud VMs. Use one of the options below:

### Option A (Recommended on cloud): Python venv + pip (GPU)

```bash
python3 -m venv ~/venvs/graphflix-pip
source ~/venvs/graphflix-pip/bin/activate
python -m pip install --upgrade pip

# Install Torch with your CUDA version (here: CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Verify GPU
python - <<'PY'
import torch; print(torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY

# Install PyG wheels matching your Torch/CUDA
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html   # use 'pt25cu121' (or your exact tag)

# Project deps
pip install -r requirements.txt

