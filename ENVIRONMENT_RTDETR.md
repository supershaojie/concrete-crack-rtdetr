# RT-DETR reproducible environment

Conda environment: rtdetr

Core versions:
- Python: 3.10.13
- NumPy: 1.26.4
- PyTorch: 2.1.2+cu121
- Torchvision: 0.16.2+cu121
- Torchaudio: 2.1.2+cu121
- OpenCV: 4.10.0.84
- Ultralytics: 8.4.21, installed from local repository with pip install -e .
- ultralytics-thop: 2.1.6
- CUDA runtime: 12.1

Project Ultralytics path:
ultralytics-main/

Installation:
1. Create conda env with Python 3.10.13
2. Install torch 2.1.2 + cu121
3. Pin numpy 1.26.4 and opencv-python 4.10.0.84
4. Run pip install -e . from ultralytics-main
