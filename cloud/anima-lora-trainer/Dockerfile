FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INPUT=1 \
    MODEL_ROOT=/models \
    SD_SCRIPTS_ROOT=/opt/sd-scripts \
    TRAINER_JOB_ROOT=/workspace/rife60-anima-jobs

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git /opt/sd-scripts
WORKDIR /opt/sd-scripts
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt
COPY app.py /app/app.py

RUN mkdir -p /models /workspace/rife60-anima-jobs
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
