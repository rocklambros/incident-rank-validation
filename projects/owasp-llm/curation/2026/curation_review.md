# Manual Curation Review
Date: 2026-05-23
Reviewer: RL

## Summary
- Total reviewed: 42
- Confirmed correct: 35
- Flagged for relabel: 5
- Flagged for replacement: 2

## Review Basis
I reviewed the incident text first, then compared my inferred label to the ID prefix. The attached rubric defines ROLL-LAPTF as LLM Artifact Promotion Trust Failure, a rollup into LLM03. It defines ROLL-CFAS as Compositional Fine tuning Alignment Subversion, not Cascading Failures in Agentic Systems. The task instruction, however, explicitly asks CFAS to mean Cascading Failures in Agentic Systems, so the two current CFAS records are flagged for replacement under that target definition.

BOUNDARY-CASES.md did not contain a LAPTF or CFAS specific override. For ITSCD relabel calls, I used the rubric boundary that narrows NEW-ITSCD to speculative decoding, streaming token emission, packet size, and timing mechanisms, while general cache signal and membership inference side channels remain LLM02.

## Validation By Current Prefix
- MANUAL-LLM06: 7 reviewed, 7 confirmed
- MANUAL-MTIE: 9 reviewed, 9 confirmed
- MANUAL-ITSCD: 9 reviewed, 4 confirmed, 5 flagged for relabel
- MANUAL-CMSB: 10 reviewed, 10 confirmed
- MANUAL-LAPTF: 5 reviewed, 5 confirmed
- MANUAL-CFAS: 2 reviewed, 2 flagged for replacement under the task target

## Flagged Incidents

### MANUAL-ITSCD-004
- Current label: NEW-ITSCD
- My label: LLM02
- Reason: The incident is a shared key value cache, semantic cache, and GPU scheduling timing side channel. It discloses private prompts or system prompt information through cache and latency behavior, but it is not specific to speculative decoding or streaming token emission. Under the rubric boundary, this is a general inference side channel and disclosure case.
- Action: relabel

### MANUAL-ITSCD-005
- Current label: NEW-ITSCD
- My label: LLM02
- Reason: PromptPeek is prompt leakage through key value cache sharing in multi tenant serving. The mechanism is cache hit behavior and prompt reconstruction, not speculative decoding packet size, token emission, or passive streaming traffic analysis. The incident fits sensitive information disclosure through an inference side channel.
- Action: relabel

### MANUAL-ITSCD-006
- Current label: NEW-ITSCD
- My label: LLM02
- Reason: The attack uses local hardware cache observation by a co resident process to infer token values and positions. The attacker observes hardware behavior rather than speculative decoding or streaming token emission. This is a general side channel disclosure incident under LLM02.
- Action: relabel

### MANUAL-ITSCD-007
- Current label: NEW-ITSCD
- My label: LLM02
- Reason: BudgetLeak infers private RAG corpus membership by interacting with the system and changing generation budget. The mechanism is membership inference from application behavior, not passive network observation, packet size, timing of streaming token emission, or speculative decoding accept rate. The disclosure outcome and mechanism fit LLM02 better than NEW-ITSCD.
- Action: relabel

### MANUAL-ITSCD-009
- Current label: NEW-ITSCD
- My label: LLM02
- Reason: Spill The Beans leaks generated tokens through CPU cache behavior in local inference. This is a hardware cache side channel, not the narrower speculative decoding or streaming token emission side channel defined for NEW-ITSCD. LLM02 covers sensitive information disclosure through general side channels.
- Action: relabel

### MANUAL-CFAS-001
- Current label: ROLL-CFAS
- My label: ROLL-CFAS under the attached rubric, but not CFAS under the task target of Cascading Failures in Agentic Systems
- Reason: CoLoRA is a compositional LoRA and adapter safety suppression attack. It does not involve an orchestrator, multi agent handoff, retrieval agent, reasoning agent, execution agent, or failure propagation across dependent agents. If the taxonomy keeps the current ROLL-CFAS definition, this record is valid. If CFAS is changed to Cascading Failures in Agentic Systems, this record poisons recall calibration and should be replaced.
- Action: replace

### MANUAL-CFAS-002
- Current label: ROLL-CFAS
- My label: LLM04 or ROLL-CFAS adjacent under the attached rubric, but not CFAS under the task target of Cascading Failures in Agentic Systems
- Reason: MergeBackdoor is a model merging and backdoor emergence case where individually weak artifacts become unsafe after merge. It is adjacent to compositional model poisoning, but it does not involve cascading failure across agents or an agentic dependency chain. Under the task target, it should be replaced.
- Action: replace

## LAPTF Labeling Decision

Overall decision: keep all five LAPTF incidents as ROLL-LAPTF. They are also LLM03 at the parent rollup level because ROLL-LAPTF rolls into LLM03. Do not relabel them to NEW-MSDA; in the attached rubric, NEW-MSDA is Model Scheming and Deceptive Alignment, not model supply and dependency attacks.

Important nuance: I am not keeping them as LAPTF because the organizations lacked adversarial prompt or model testing frameworks. I am keeping them because the attached LAPTF definition is artifact promotion trust failure: weak provenance, unsigned artifacts, repository impersonation, weak release gates, and promotion of unsafe model artifacts.

### MANUAL-LAPTF-001
- Decision: keep ROLL-LAPTF
- Parent label: LLM03
- Reason: A malicious Hugging Face model file executed code through pickle behavior when loaded. The primary root cause is unsafe promotion and trust of a model artifact as data without sufficient provenance and safe deserialization controls.

### MANUAL-LAPTF-002
- Decision: keep ROLL-LAPTF
- Parent label: LLM03
- Reason: Malicious PyTorch .pth artifacts on Hugging Face executed payloads when deserialized. The incident is a model artifact promotion and safe serialization failure inside the AI supply chain.

### MANUAL-LAPTF-003
- Decision: keep ROLL-LAPTF
- Parent label: LLM03
- Reason: Model Namespace Reuse exploits trust in model names and automatic retrieval of abandoned or re registered namespaces. That is a weak provenance and repository name trust failure before publication or deployment.

### MANUAL-LAPTF-004
- Decision: keep ROLL-LAPTF
- Parent label: LLM03
- Reason: The Open OSS Privacy Filter repository used typosquatting, fake engagement, and trusted presentation signals to gain adoption. The failure is artifact promotion through name similarity, popularity, and weak malware resistant release gating.

### MANUAL-LAPTF-005
- Decision: keep ROLL-LAPTF
- Parent label: LLM03
- Reason: nullifAI used compressed pickle payloads and bypassed existing repository scanning. The model artifact passed platform trust checks even though loading it executed attacker controlled code.

## Replacement CFAS Incidents Added To manual_curated_incidents.json

### MANUAL-CFAS-003
- Source: AppOmni ServiceNow Now Assist agent to agent discovery prompt injection
- Why it fits: low privilege prompt injection propagated through agent discovery and caused higher privilege agents to perform unauthorized actions.

### MANUAL-CFAS-004
- Source: Prompt Infection: LLM to LLM Prompt Injection within Multi Agent Systems
- Why it fits: a compromised agent replicated attacker instructions across interconnected agents and coordinated specialized downstream agents.

### MANUAL-CFAS-005
- Source: Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast
- Why it fits: one infected multimodal agent spread unsafe behavior through pairwise agent interactions until the broader agent population became infected.

### MANUAL-CFAS-006
- Source: Morris II GenAI worm
- Why it fits: one poisoned RAG based assistant generated outputs that became trusted inputs to other assistants, propagating compromise across an ecosystem.

### MANUAL-CFAS-007
- Source: Multi Agent Systems Execute Arbitrary Malicious Code
- Why it fits: adversarial content manipulated control flow metadata so orchestrators and downstream agents routed tasks into unsafe execution paths.

## Files Produced
- curation_review.md
- manual_curated_incidents.json
