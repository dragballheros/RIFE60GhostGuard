# Thin overlay on RunPod's published ComfyUI worker image. This avoids rebuilding
# ComfyUI and CUDA dependencies while adding the current model-folder mappings used
# by ANIMA's UNETLoader/CLIPLoader. Update the pinned upstream tag deliberately.
FROM runpod/worker-comfyui:5.10.0-base
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
