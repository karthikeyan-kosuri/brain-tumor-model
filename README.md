##  Installation

### 1. Clone the repository
```bash
git clone https://github.com/karthikeyan-kosuri/brain-tumor-model
cd brain-tumor-model
```
### 2. Create a Virtual environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```
### 3. Install dependencies
1. CPU-only:
```bash
pip install -r requirements.txt
````
2. Cuda:
```bash
# Example: CUDA 12.8 build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
