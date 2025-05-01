# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/comfyui/essentials/essentials_example.py"]
# ---

import subprocess
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")  # start from basic Linux with Python
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("comfy-cli==1.2.7")  # install comfy-cli
    .run_commands(  # use comfy-cli to install the ComfyUI repo and its dependencies
        "comfy --skip-prompt install --nvidia"
    )
    .run_commands(  # download the ComfyUI Essentials custom node pack
        "comfy node install ComfyUI_essentials",
        # "comfy node install ComfyUI-KJNodes",
        # "comfy node install ComfyUI-VideoHelperSuite",
        # "comfy node install ComfyUI-WanVideoWrapper",
        # "comfy node install ComfyUI-Crystools",
    )
    .run_commands(  # download the ComfyUI custom node pack
        "apt-get update && apt-get install ffmpeg libsm6 libxext6  -y"
    )
    .run_commands(  # 下载 Custom Node
        "git clone https://github.com/crystian/ComfyUI-Crystools /root/comfy/ComfyUI/custom_nodes/ComfyUI-Crystools",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-Crystools",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-Crystools/requirements.txt",
        "git clone https://github.com/kijai/ComfyUI-KJNodes /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt",
        "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt",
        "git clone https://github.com/kijai/ComfyUI-WanVideoWrapper /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt",
        "git clone https://github.com/cubiq/ComfyUI_essentials /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
        "cd /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials/requirements.txt",
    )
    # .run_commands(
    #     "comfy --skip-prompt node install https://github.com/crystian/ComfyUI-Crystools",
    #     "comfy --skip-prompt node install https://github.com/kijai/ComfyUI-KJNodes",
    #     "comfy --skip-prompt node install https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    #     "comfy --skip-prompt node install https://github.com/kijai/ComfyUI-WanVideoWrapper",
    # )
    .run_commands(  # 下载 unet 模型
        "comfy --skip-prompt model download --url https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Skyreels/Wan2_1-SkyReels-V2-DF-1_3B-540P_fp32.safetensors --relative-path models/unet"
    )
    .run_commands(  # 下载 VAE 模型
        "comfy --skip-prompt model download --url https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_fp32.safetensors --relative-path models/vae"
    )
    .run_commands(  # 下载 clip 模型
        "comfy --skip-prompt model download --url https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/open-clip-xlm-roberta-large-vit-huge-14_visual_fp32.safetensors --relative-path models/clip"
    )
    .run_commands(  # 下载 text_encoders 模型
        "comfy --skip-prompt model download --url https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp16.safetensors --relative-path models/text_encoders"
    )
)


app = modal.App(name="example-essentials-comfyui", image=image)


# Run ComfyUI as an interactive web server
@app.function(
    max_containers=1,
    scaledown_window=30,
    timeout=1800,
    gpu="L40S",
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
