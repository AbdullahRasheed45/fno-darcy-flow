FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Single GPU:  docker run --gpus 1 <img> python -m src.train --data data/darcy_train.npz
# Multi GPU:   docker run --gpus all <img> torchrun --standalone --nproc_per_node=2 -m src.train --data data/darcy_train.npz
CMD ["python", "-m", "pytest", "-q"]
