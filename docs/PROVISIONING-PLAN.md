# Provisioning plan (Plan 5 cycle)

## GPU provider selection (HANDOFF v2.5 section 7.5)

**Default rule:** Use RunPod for any Stage-2 GPU workload that cannot complete on the local Jetson GPU in under 30 minutes wall time. Below 30 minutes the local Jetson wins on end-to-end latency (no upload, auth, or weight-transfer overhead). Above 30 minutes the RunPod per-iteration speed advantage (H100 / A100) dominates and the cost is worth it. The rule is per-workload, not per-cycle.

| Workload class | Typical scale | Provider | Why |
|---|---|---|---|
| Full Stage-2 cycle classification (70B-class model) | ~7,000 incidents | **RunPod (REQUIRED)** | >50h on Jetson; >30 min threshold trivially met |
| Stage-2 cycle with 8B-class model | ~7,000 incidents | **RunPod (REQUIRED)** | >10h on Jetson |
| Ad-hoc adjudication batch | <200 incidents | local Jetson | typically <30 min |
| Embedding-based rubric clustering | <5,000 vectors | local Jetson | typically <30 min |
| Single-rule spot check | 10-50 incidents | local Jetson | typically <30 min |

**Decision procedure (mechanical):**
1. Estimate local Jetson wall time: (tokens/sec) x (tokens/incident) x (incidents). Commit estimate to `cycle/provenance/local_run_estimate.json` BEFORE the workload starts.
2. If estimated wall time < 30 min, run local. Record the actual wall time on completion.
3. If estimated wall time >= 30 min, provision RunPod per the rest of this plan.
4. If estimate is wrong by >= 2x during execution, abort and re-provision on RunPod. Log the misestimate as a post-hoc analysis (HANDOFF section 6 control 11(f), Merkle-chained per M16).

**No CPU-bound workloads are GPU-provisioned regardless** (HANDOFF v2.5 section 7.5): NUTS, vote bootstrap, twin, predictive sampling stay on CPU for reproducibility.

## Cost

- **Per-cycle ceiling: $500 USD** (default per M9; override here with explicit Rock authorization)
- Per-hour budget: <TBD>
- Monitoring: RunPod billing API polled every 10 min; auto-shutoff at 1.2x ceiling
- Override authorization: if ceiling > $500, REVIEWERS.md PRE-PUBLISH CHECKLIST must include "cost ceiling exceeded by $X, authorized by <name> on <date>"
- Local Jetson runs: $0 GPU cost (electricity ignored); only counted in cost ceiling if a RunPod fallback fires after a misestimate

## GPU (RunPod, when triggered)

- Provider: RunPod
- GPU type: <TBD before Plan 5 cycle — prefer H100 80GB or H200 if available>
- GPU count: <TBD — maximize parallelism within cost ceiling>
- Region: <TBD — prefer US East for latency to model weights>

## Model

- Model identity: <TBD before Plan 5 — e.g., meta-llama/Llama-3.1-70B-Instruct>
- Weight provenance hash: <SHA-256 of weight checkpoint, captured at cycle start>
- Determinism: temperature=0, top_p=1.0, seed=<from PreregManifest.prng_seed>

## Workload

- Input: ambiguous incidents from corpus A (gold-set tagged)
- Batch size: <TBD — balance throughput vs per-batch determinism>
- Expected wall time: <TBD — target under per-cycle ceiling>

## Outputs

- Stage-2 assignments JSONL: cycle/results/stage2_assignments.jsonl
- Stage-2 provenance: cycle/results/stage2_provenance.json (model_identity, weight_provenance_hash, prng_seed, batch_size, wall_time, cost, provider="runpod"|"local-jetson")
- Hashes committed to PreregManifest before `decide` phase

## Reproducibility

- Pinned model version (no auto-upgrade)
- Pinned weight checkpoint hash
- Deterministic decoding (temperature=0, seed-pinned)
- Batch determinism verified by re-running a 10-incident sample twice and asserting identical Assignment outputs
- Cross-provider determinism: if a workload migrates from local Jetson to RunPod mid-cycle (after a misestimate abort), the entire workload re-runs on RunPod — partial-results carry-over is forbidden
