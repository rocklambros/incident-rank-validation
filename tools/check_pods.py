#!/usr/bin/env python3
"""Check health of deployed RunPod vLLM pods."""
import json
import sys

import httpx

PODS = json.loads(open("tools/runpod_pods.json").read())


def check_pod(pod: dict) -> dict:
    proxy_url = f"https://{pod['pod_id']}-8000.proxy.runpod.net"
    status = {"name": pod["name"], "url": proxy_url}
    try:
        resp = httpx.get(f"{proxy_url}/v1/models", timeout=15.0)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            model_ids = [m["id"] for m in models]
            status["status"] = "READY"
            status["models"] = model_ids
        else:
            status["status"] = f"HTTP {resp.status_code}"
    except httpx.ConnectError:
        status["status"] = "NOT_READY (connection refused)"
    except httpx.ReadTimeout:
        status["status"] = "NOT_READY (timeout — model loading)"
    except Exception as e:
        status["status"] = f"ERROR: {e}"
    return status


def main():
    all_ready = True
    for pod in PODS:
        result = check_pod(pod)
        marker = "OK" if result.get("status") == "READY" else "  "
        print(f"[{marker}] {result['name']:20s} {result['status']}")
        if result.get("models"):
            for m in result["models"]:
                print(f"     model: {m}")
        if result.get("status") != "READY":
            all_ready = False

    if all_ready:
        print("\nAll pods READY.")
    else:
        print("\nSome pods still loading. Check RunPod console.")
    return 0 if all_ready else 1


if __name__ == "__main__":
    sys.exit(main())
