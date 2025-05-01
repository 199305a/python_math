from pathlib import Path

import modal.volume

import modal


GPU_CONFIG = "L40S:4"
app = modal.App("deepseek_brid")


@app.local_entrypoint()
def main(
    prompt: str = None,
    model: str = "DeepSeek-R1",
    n_predict: int = -1,
    args: str = None,
):
    import shlex

    org_name = "unsloth"
    if model.lower() == "phi-4":
        model_name = "phi-4-GGUF"
        quant = "Q2_K"
        model_entrypoint_file = f"phi-4-{quant}.gguf"
        model_pattern = f"*{quant}*"
        revision = None
        if args is None:
            args = DEFAULT_PHI_ARGS
        else:
            args = shlex.split(args)
    elif model.lower() == "deepseek-r1":
        model_name = "DeepSeek-R1-GGUF"
        quant = "UD-IQ1_S"
        model_entrypoint_file = (
            f"{model}-{quant}/DeepSeek-R1-{quant}-00001-of-00003.gguf"
        )
        model_pattern = f"*{quant}*"
        revision = "02656f62d2aa9da4d3f0cdb34c341d30dd87c3b6"
        if args is None:
            args = DEFAULT_DEEPSEEK_R1_ARGS
        else:
            args = shlex.split(args)
    else:
        raise ValueError(f"Unknown model: {model}")

    repo_id = f"{org_name}/{model_name}"

    download_model.remote(repo_id, [model_pattern], revision)

    result = llama_cpp_inference.remote(
        model_entrypoint_file,
        prompt=prompt,
        n_predict=n_predict,
        args=args,
        store_output=model.lower() == "deepseek-r1",
    )

    output_path = Path("/tmp") / f"llama-cpp-{model}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing output to {output_path}")
    output_path.write_text(result)


DEFAULT_DEEPSEEK_R1_ARGS = [  # good default llama.cpp cli args for deepseek-r1
    "--cache-type-k",
    "q4_0",
    "--threads",
    "12",
    "-no-cnv",
    "--prio",
    "2",
    "--temp",
    "0.6",
    "--ctx-size",
    "8192",
]
DEFAULT_PHI_ARGS = [  # good default llama.cpp cli args for phi-4
    "--threads",
    "16",
    "-no-cnv",
    "--ctx-size",
    "16384",
]

LLAMA_CPP_RELEASE = "b4568"
MINUTES = 60

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    .run_commands("git clone https://github.com/ggerganov/llama.cpp")
    .run_commands(
        "cmake llama.cpp -B llama.cpp/build "
        "-DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON "
    )
    .run_commands(  # this one takes a few minutes!
        "cmake --build llama.cpp/build --config Release -j --clean-first --target llama-quantize llama-cli"
    )
    .run_commands("cp llama.cpp/build/bin/llama-* llama.cpp")
    .entrypoint([])  # remove NVIDIA base container entrypoint
)

model_cache = modal.Volume.from_name("model-cache", create_if_missing=True)
cache_dir = "/root/.cache/llama.cpp"

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=download_image, volumes={cache_dir: model_cache}, timeout=30 * MINUTES
)
def download_model(repo_id, allow_patterns, revision: str = None):

    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} ")
    snapshot_download(
        repo_id, local_dir=cache_dir, allow_patterns=allow_patterns, revision=revision
    )
    model_cache.commit()
    print(f"Downloaded {repo_id} to {cache_dir}")


results = modal.Volume.from_name("llamacpp-results", create_if_missing=True)
results_dir = "/root/results"


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={results_dir: results, cache_dir: model_cache},
    timeout=30 * MINUTES,
)
def llama_cpp_inference(
    model_entrypoint_file,
    prompt: str = None,
    n_predict: int = -1,
    args: list[str] = None,
    store_output: bool = True,
):
    import subprocess
    from uuid import uuid4

    if prompt is None:
        prompt = DEFAULT_PROMPT
    if "deepseek" in model_entrypoint_file.lower():
        prompt = "<|User|>" + prompt + "<think>"
    if args is None:
        args = []

    if GPU_CONFIG is not None:
        n_gpu_layers = 9999
    else:
        n_gpu_layers = 0

    if store_output:
        result_id = str(uuid4())
        print(f"Storing output in {results_dir}/{result_id}")

    command = [
        "/llama.cpp/llama-cli",
        "--model",
        f"{cache_dir}/{model_entrypoint_file}",
        "--n-gpu-layers",
        str(n_gpu_layers),
        "--prompt",
        prompt,
        "--n-predict",
        str(n_predict),
    ] + args
    print(f"Running {command}")
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False
    )
    stdout, stderr = collect_output(p)
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, command, stdout, stderr)

    if store_output:
        print(f"Writing output to {results_dir}/{result_id}.txt")

        result_dir = Path(results_dir) / result_id
        result_dir.mkdir(parents=True)
        (result_dir / "out.txt").write_text(stdout)
        (result_dir / "err.txt").write_text(stderr)

    return stdout


DEFAULT_PROMPT = """Create a Flappy Bird game in Python. You must include these things:

    You must use pygame.
    The background color should be randomly chosen and is a light shade. Start with a light blue color.
    Pressing SPACE multiple times will accelerate the bird.
    The bird's shape should be randomly chosen as a square, circle or triangle. The color should be randomly chosen as a dark color.
    Place on the bottom some land colored as dark brown or yellow chosen randomly.
    Make a score shown on the top right side. Increment if you pass pipes and don't hit them.
    Make randomly spaced pipes with enough space. Color them randomly as dark green or light brown or a dark gray shade.
    When you lose, show the best score. Make the text inside the screen. Pressing q or Esc will quit the game. Restarting is pressing SPACE again.

The final game should be inside a markdown section in Python. Check your code for errors and fix them before the final markdown section."""


def stream_output(stream, queue, write_stream):
    for line in iter(stream.readline, b""):
        line = line.decode("utf-8", errors="replace")
        write_stream.write(line)
        write_stream.flush()
        queue.put(line)

    stream.close()


def collect_output(process):
    import sys
    from queue import Queue
    from threading import Thread

    stdout_queue = Queue()
    stderr_queue = Queue()
    stdout_thread = Thread(
        target=stream_output, args=(process.stdout, stdout_queue, sys.stdout)
    )
    stderr_thread = Thread(
        target=stream_output, args=(process.stderr, stderr_queue, sys.stderr)
    )
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()
    process.wait()
    stdout_collected = "".join(stdout_queue.queue)
    stderr_collected = "".join(stderr_queue.queue)

    return stdout_collected, stderr_collected
