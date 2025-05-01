# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/comfyui/essentials/essentials_example.py"]
# ---

import subprocess
import modal


CIVIKEY = "8bbb5825cadf3dc8c6e2a0dc8fec647c"
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
        # "git clone https://github.com/ltdrdata/ComfyUI-Manager /root/comfy/ComfyUI/custom_nodes/ComfyUI-Manager",
        # "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt",
        "git clone https://github.com/AIGODLIKE/AIGODLIKE-ComfyUI-Translation /root/comfy/ComfyUI/custom_nodes/AIGODLIKE-ComfyUI-Translation",
        # "pip install -r /root/comfy/ComfyUI/custom_nodes/AIGODLIKE-ComfyUI-Translation/requirements.txt",
        "git clone https://github.com/chrisgoringe/cg-use-everywhere /root/comfy/ComfyUI/custom_nodes/cg-use-everywhere",
        # "pip install -r /root/comfy/ComfyUI/custom_nodes/cg-use-everywhere/requirements.txt",
        "git clone https://github.com/rgthree/rgthree-comfy /root/comfy/ComfyUI/custom_nodes/rgthree-comfy",
        # "git clone https://github.com/rgthree/rgthree-comfy /root/comfy/ComfyUI/custom_nodes/rgthree-comfy",
        # "pip install -r /root/comfy/ComfyUI/custom_nodes/rgthree-comfy/requirements.txt",
        "git clone https://github.com/chengzeyi/Comfy-WaveSpeed /root/comfy/ComfyUI/custom_nodes/Comfy-WaveSpeed",
        "git clone https://github.com/crystian/ComfyUI-Crystools /root/comfy/ComfyUI/custom_nodes/ComfyUI-Crystools",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-Crystools/requirements.txt",
        "git clone https://github.com/kijai/ComfyUI-HunyuanVideoWrapper /root/comfy/ComfyUI/custom_nodes/ComfyUI-HunyuanVideoWrapper",
        "pip install -r /root/comfy/ComfyUI/custom_nodes/ComfyUI-HunyuanVideoWrapper/requirements.txt",
    )
    # .run_commands(
    #     "comfy --skip-prompt node install https://github.com/crystian/ComfyUI-Crystools",
    #     "comfy --skip-prompt node install https://github.com/kijai/ComfyUI-KJNodes",
    #     "comfy --skip-prompt node install https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    #     "comfy --skip-prompt node install https://github.com/kijai/ComfyUI-WanVideoWrapper",
    # )
    .run_commands(  # 下载 unet 模型
        "comfy --skip-prompt model download --url https://huggingface.co/Kijai/HunyuanVideo_comfy/resolve/main/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors?download=true --relative-path models/unet"
    )
    .run_commands(  # 下载 VAE 模型
        "comfy --skip-prompt model download --url https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/vae/hunyuan_video_vae_bf16.safetensors?download=true --relative-path models/vae"
    )
    .run_commands(  # 下载 clip 模型
        "comfy --skip-prompt model download --url https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/clip_l.safetensors?download=true --relative-path models/clip",
        "comfy --skip-prompt model download --url https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/text_encoders/llava_llama3_fp8_scaled.safetensors?download=true --relative-path models/clip",
    )
    .run_commands(  # 下载 text_encoders 模型
        # "comfy --skip-prompt model download --url https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors --relative-path models/text_encoders"
        "comfy --skip-prompt model download --url https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth?download=true --relative-path models/upscale_models"
    )
    .run_commands(  # 下载 lora 模型
        # "mkdir -p models/loras",
        # "curl -o models/loras/4x_foolhardy_Remacri.pth https://civitai.com/api/download/models/1274571?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c",
        # "curl -o models/loras/4x_foolhardy_Remacri.pth https://civitai.com/api/download/models/1273832?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c",
        # "curl -o models/loras/4x_foolhardy_Remacri.pth https://civitai.com/api/download/models/1184914?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c",
        # "curl -o models/loras/4x_foolhardy_Remacri.pth https://civitai.com/api/download/models/1282806?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c",
        # "curl -o models/loras/4x_foolhardy_Remacri.pth https://civitai.com/api/download/models/1197482?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c",
        # "curl -o models/loras/4x_foolhardy_Remacri.pth https://civitai.com/api/download/models/1274571?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c",
        "comfy --skip-prompt model download --url https://civitai.com/api/download/models/1274571?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c --relative-path models/loras",
        "comfy --skip-prompt model download --url https://civitai.com/api/download/models/1273832?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c --relative-path models/loras",
        "comfy --skip-prompt model download --url https://civitai.com/api/download/models/1184914?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c --relative-path models/loras",
        "comfy --skip-prompt model download --url https://civitai.com/api/download/models/1282806?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c --relative-path models/loras",
        "comfy --skip-prompt model download --url https://civitai.com/api/download/models/1197482?type=Model&format=SafeTensor&token=8bbb5825cadf3dc8c6e2a0dc8fec647c --relative-path models/loras",
    )
)


app = modal.App(name="example-essentials", image=image)


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
