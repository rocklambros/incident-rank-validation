"""Stage-2 LLM classifier manifest — pinned model, prompt, and execution provenance."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Stage2Manifest:
    model_identity: str
    weight_provenance_hash: str
    prompt_hash: str
    prompt_template_version: str
    batch_size: int
    prng_seed: int
    temperature: float
    top_p: float
    cost_ceiling_usd: float
    provider: str
    gpu_type: str | None
    gpu_count: int | None
    region: str | None
    runpod_job_ids: tuple[str, ...]
    wall_time_seconds: float | None
    actual_cost_usd: float | None
    incidents_classified: int | None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, tuple):
                d[f.name] = list(v)
            else:
                d[f.name] = v
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def read(cls, path: Path) -> Stage2Manifest:
        d = json.loads(path.read_text())
        if "runpod_job_ids" in d and isinstance(d["runpod_job_ids"], list):
            d["runpod_job_ids"] = tuple(d["runpod_job_ids"])
        return cls(**d)
