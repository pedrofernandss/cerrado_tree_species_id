# 🌳 Cerrado Tree Identification

Welcome to the **cerrado-tree-id** project!

This repository introduces a novel **image preprocessing pipeline** that fuses multispectral images, tailored for native species of the Cerrado biome.  
Using our local dataset, the models provided in `detect/models` can detect **over 16 different species** — feel free to use them as you like!

Our pipeline delivers a **3.0% improvement in detection accuracy** compared to state-of-the-art RGB-only approaches.

> ⚠️ **Note:** Running the preprocessing notebooks requires a machine with a **GPU** for efficient processing.

---

## 🚀 Getting Started

Follow the steps below to set up and run the project:

### 1. Create a virtual environment

We recommend using `venv` or `conda` to manage dependencies.

```bash
python -m venv your_env
source your_env/bin/activate
````

### 2. Install dependencies

* Install `exiftool` for extracting image metadata:

```bash
sudo apt install libimage-exiftool-perl
```

* Then, install Python packages:

```bash
pip install -r requirements.txt
```

---

## ⚙️ CUDA Setup (GPU Acceleration)

To run preprocessing efficiently, you’ll need a working CUDA environment. Here’s how to set it up:

### 1. Check for a compatible NVIDIA GPU

```bash
nvidia-smi
```

If your GPU appears in the output, you're good to go!

### 2. Install NVIDIA Drivers

Download and install the latest drivers for your GPU from:
[https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

### 3. Install CUDA Toolkit and cuDNN

Choose a version compatible with your PyTorch installation. Example for CUDA 11.8:

```bash
# Ubuntu example
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y cuda
```

After installation, add CUDA to your path:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 4. Verify CUDA is working in Python

```python
import torch
print(torch.cuda.is_available())  # Should return True
```

---

## 🧪 Running the Pipeline

After a functional GPU and an appropriate environment are set up, you're ready to run the project!

### Step 1: Preprocess images

Open and run the notebook:

```
./preprocessing/preprocess_imgs.ipynb
```

🖼️ Preprocessed images will be saved to:

```
./preprocessing/preprocessed-imgs
```

### Step 2: Fuse multispectral channels

Open and run the notebook:

```
./preprocessing/fuse_imgs.ipynb
```

🔀 Fused images will be saved to:

```
./preprocessing/fused-imgs
```

🖼️ Corresponding RGB images are also stored in:

```
./preprocessing/rgb-imgs
```

### Step 3: Run detection on fused images

Open and run the notebook:

```
./detect/demo.ipynb
```

📁 Detection results will be saved to:

```
./runs/detect/predict
```

---

✅ That’s it! You're ready to explore tree species detection in the Cerrado using fused multispectral data.
