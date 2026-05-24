#!/usr/bin/env python3
"""Deploy 3 vLLM model pods on RunPod H200 GPUs via REST API.

Uses dockerStartCmd with ["/bin/bash", "-c", "cmd"] format to ensure
shell command parsing works correctly with RunPod's container runtime.
"""
import json
import os
import subprocess
import sys

import httpx


def load_secret(pass_name: str, env_var: str) -> str:
    val = os.environ.get(env_var, "")
    if val:
        return val
    result = subprocess.run(
        ["pass", "show", pass_name], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


MODELS = [
    {
        "name": "qwen3-235b",
        "model_id": "Qwen/Qwen3-235B-A22B",
        "gpu_type": "NVIDIA H200",
        "gpu_count": 4,
        "container_disk_gb": 300,
        "vllm_cmd": (
            "vllm serve Qwen/Qwen3-235B-A22B "
            "--host 0.0.0.0 --port 8000 "
            "--tensor-parallel-size 4 "
            "--enable-expert-parallel "
            "--max-model-len 4096 "
            "--gpu-memory-utilization 0.90 "
            "--trust-remote-code"
        ),
    },
    {
        "name": "llama-405b",
        "model_id": "meta-llama/Llama-3.1-405B-Instruct-FP8",
        "gpu_type": "NVIDIA H200",
        "gpu_count": 4,
        "container_disk_gb": 500,
        "vllm_cmd": (
            "vllm serve meta-llama/Llama-3.1-405B-Instruct-FP8 "
            "--host 0.0.0.0 --port 8000 "
            "--tensor-parallel-size 4 "
            "--max-model-len 4096 "
            "--gpu-memory-utilization 0.90 "
            "--allow-deprecated-quantization "
            "--trust-remote-code"
        ),
    },
    {
        "name": "deepseek-v3",
        "model_id": "deepseek-ai/DeepSeek-V3",
        "gpu_type": "NVIDIA H200",
        "gpu_count": 8,
        "container_disk_gb": 800,
        "vllm_cmd": (
            "vllm serve deepseek-ai/DeepSeek-V3 "
            "--host 0.0.0.0 --port 8000 "
            "--tensor-parallel-size 8 "
            "--enable-expert-parallel "
            "--max-model-len 4096 "
            "--gpu-memory-utilization 0.90 "
            "--trust-remote-code"
        ),
    },
]

IMAGE = "vllm/vllm-openai:v0.21.0"
REST_BASE = "https://rest.runpod.io/v1"


def create_pod_rest(
    api_key: str,
    name: str,
    image: str,
    gpu_type: str,
    gpu_count: int,
    container_disk_gb: int,
    vllm_cmd: str,
    env: dict,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeIds": [gpu_type],
        "gpuCount": gpu_count,
        "containerDiskInGb": container_disk_gb,
        "volumeInGb": 0,
        "ports": ["8000/http", "22/tcp"],
        "dockerEntrypoint": ["/bin/bash"],
        "dockerStartCmd": ["-c", vllm_cmd],
        "env": env,
        "supportPublicIp": True,
    }
    resp = httpx.post(
        f"{REST_BASE}/pods",
        headers=headers,
        json=payload,
        timeout=30.0,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Pod creation failed ({resp.status_code}): {resp.text}")
    return resp.json()


def main():
    api_key = load_secret("runpod/api-key", "RUNPOD_API_KEY")
    hf_token = load_secret("huggingface/token", "HF_TOKEN")

    pods_created = []

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Deploying {model['name']}...")
        print(f"  Model: {model['model_id']}")
        print(f"  GPUs:  {model['gpu_count']}x {model['gpu_type']}")
        print(f"  Disk:  {model['container_disk_gb']}GB")
        print(f"{'='*60}")

        env = {
            "HF_TOKEN": hf_token,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }

        pod_name = f"classify-{model['name']}"
        pod = create_pod_rest(
            api_key=api_key,
            name=pod_name,
            image=IMAGE,
            gpu_type=model["gpu_type"],
            gpu_count=model["gpu_count"],
            container_disk_gb=model["container_disk_gb"],
            vllm_cmd=model["vllm_cmd"],
            env=env,
        )
        pod_id = pod["id"]
        print(f"  Pod created: {pod_id}")
        pods_created.append({
            "name": model["name"],
            "model_id": model["model_id"],
            "pod_id": pod_id,
        })

    print(f"\n{'='*60}")
    print("ALL PODS CREATED")
    print(f"{'='*60}")
    for p in pods_created:
        proxy_url = f"https://{p['pod_id']}-8000.proxy.runpod.net"
        print(f"  {p['name']:20s} pod={p['pod_id']}  url={proxy_url}")

    out_path = "tools/runpod_pods.json"
    with open(out_path, "w") as f:
        json.dump(pods_created, f, indent=2)
    print(f"\nPod info saved to {out_path}")

    print(f"\n{'='*60}")
    print("ENVIRONMENT VARIABLES FOR RECLASSIFY")
    print(f"{'='*60}")
    for i, p in enumerate(pods_created, 1):
        proxy_url = f"https://{p['pod_id']}-8000.proxy.runpod.net"
        print(f"export RUNPOD_MODEL_{i}_URL='{proxy_url}'")
        print(f"export RUNPOD_MODEL_{i}_NAME='{p['model_id']}'")

    print("\nModel download + load takes 15-45 min. Monitor: https://www.runpod.io/console/pods")


if __name__ == "__main__":
    main()
