# 🌳 Cerrado Tree Species Identificator

Welcome to the **Cerrado Tree Species Identificator** project!

This repository introduces a novel **image preprocessing pipeline** that fuses multispectral images, tailored for native species of the Cerrado biome.  
Using our local dataset, the models provided in `models` can detect **over 18 different species**. Feel free to use them as you like!

Our pipeline delivers a **4.2% improvement in detection accuracy** compared to state-of-the-art RGB-only approaches.

> **Note:** Running the preprocessing notebooks requires a machine with a **GPU** for efficient processing.

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
sudo apt update && sudo apt upgrade 
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

Open and run the python script:

```
./src/preprocessing/main.py
```

Preprocessed images will be saved to:

```
{path-to-dataset}/{specified-date}
```

### Step 2: Fuse multispectral channels

Open and run the python script:

```
./src/image_fusion/main.py
```

Fused images will be saved to:

```
{path-to-dataset}/{specified-date}/fused-imgs
```

### Step 3: Generate the vegetation indexes and the combined images

Open and run the python script:

```
./src/vegetation_index/main.py
```

The vegetation indexes images will be saved to:

```
{path-to-dataset}/{specified-date}/ndre
{path-to-dataset}/{specified-date}/ndvi
```

The vegetation indexes images combined/stacked with the RGB and Fused images will be saved to:

```
{path-to-dataset}/{specified-date}/fused-ndre
{path-to-dataset}/{specified-date}/fused-ndvi

{path-to-dataset}/{specified-date}/rgb-ndre
{path-to-dataset}/{specified-date}/rgb-ndvi
```


### Step 4: Train the models

The models are trained in batch for all the image types and variation availabe. For train a specific model, go to `src/models/{selected-model}/train` and execute: 

```
train.sh
```

Artifacts generated during trainig such as images, weights, charts and etc. will be saved in `runs/{model-name}/no-augmentation/{image-type}`

### Step 5: Evaluate training

To evaluate the executed trainig in the test set, go to `src/models/{selected-model}/eval` and execute:

```
eval.sh
```

The artifacts generated such as images and charts will be saved in `reports/{model-name}/test_no_augmented_{image-type}`

---

That’s it! You're ready to explore tree species detection in the Cerrado using fused multispectral data.
