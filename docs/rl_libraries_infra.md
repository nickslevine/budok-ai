# RL Libraries & Infrastructure

Research notes on platforms for RL-training LLM agents to play YOMI Hustle.

## Tinker (Thinking Machines Lab)

**URL:** https://tinker-docs.thinkingmachines.ai/tinker/

### What It Is

Tinker is a **cloud-hosted, distributed LLM post-training platform**. You write ordinary Python on your local CPU machine; Tinker's GPU worker pool handles all forward/backward passes and optimizer steps remotely. Pay-per-token, no GPU infrastructure to manage.

It is **not** a general RL framework (like Gymnasium or RLlib). It is specifically designed for LLM post-training: SFT, RLVR, RLHF, and GRPO. The "environment" is always a text-in/text-out task where the model generates natural language responses.

### Architecture

- **Client-side (your machine):** CPU-only Python. Install via `uv pip install tinker`, set `TINKER_API_KEY`.
- **Server-side (Tinker's infra):** Distributed GPU pool running synchronized training cycles. Multi-tenant.
- **Training method:** LoRA only (rank 32 default). No full fine-tuning. Learning rates must be 20-100x higher than full fine-tuning.
- **Optimizer:** Adam/AdamW only.

### Client Hierarchy

| Client | Purpose |
|---|---|
| `ServiceClient` | Entry point; creates all other clients |
| `TrainingClient` | Forward/backward passes, optimizer steps, checkpoint management |
| `SamplingClient` | Text generation (inference), separate from training |
| `RestClient` | Checkpoint and run management via REST |

### Supported RL Algorithms

| Loss Function | Behavior |
|---|---|
| `importance_sampling` | Off-policy correction via p/q ratio x advantage |
| `ppo` | Clips importance ratio at epsilon=0.2 |
| `cispo` | Clips ratio, uses as stopped-gradient coefficient |
| `dro` | Quadratic penalty on KL divergence |
| `forward_backward_custom` | User-supplied Python loss function (~1.5x FLOPs) |

Primary recommended approach is **GRPO** (Group Relative Policy Optimization):
1. Save current weights, create SamplingClient
2. Generate `group_size` rollouts per prompt
3. Score with reward function
4. Normalize rewards within each group (group-relative advantage)
5. Skip degenerate groups (all same reward)
6. Build training data, forward_backward + optim_step
7. Repeat

### Environment Abstraction

```python
# Base class - multi-turn
class Env:
    def initial_observation(self) -> (prompt, stop_conditions)
    def step(self, action: str) -> (reward, done, metrics)

# Convenience class - single-turn Q&A
class ProblemEnv:
    def get_question(self) -> str
    def check_answer(self, sample_str) -> bool
    def check_format(self, sample_str) -> bool
    def get_reference_answer(self) -> str
```

Also supports `MessageEnv` for multi-turn conversations.

### Training Loop Pattern (GRPO)

```python
async def train():
    service = tinker.ServiceClient()
    tc = await service.create_lora_training_client(base_model="...", rank=32)
    
    for epoch in range(num_epochs):
        # Save weights for sampling
        sc = await tc.save_weights_and_get_sampling_client("epoch_{epoch}")
        
        for batch in dataset:
            # Generate rollouts
            groups = await do_group_rollout_and_filter_constant_reward(batch, sc)
            
            # Compute advantages (GRPO)
            advantages = compute_advantages(groups)
            
            # Train
            data = assemble_training_data(groups, advantages)
            await tc.forward_backward_async(data, loss_fn="ppo")
            await tc.optim_step_async(AdamParams(learning_rate=1e-4))
```

### Existing Environment Examples

- **Multi-Agent RL:** Tic-tac-toe self-play, guess the number, 20 questions
- **Harbor RL:** Bash-using agents in isolated containers (Terminal-Bench, SWE-Bench)
- **Agent RL:** Tool-using agents via MCP servers in Modal sandboxes
- **Math/Code:** Verification-based rewards (AIME, GSM8K, code execution)

### Models Available

Supports models from 1B to 1T+ parameters (dense and MoE). MoE models are priced by active parameters. Specific model list not publicly documented - need to check `get_server_capabilities()`.

### Relevance to Yomi Hustle

**High relevance.** Our architecture already works as text-in/text-out: game state is serialized to structured text, LLM selects a named action from legal actions, game resolves. Tinker can train the LLM policy end-to-end:
- Observation = text-serialized game state (already exists in our protocol)
- Action = named move selection from legal action set (already text)
- Reward = HP delta per turn + win/loss terminal signal (already tracked)

The GRPO loop maps naturally: generate rollouts by playing matches, score by game outcome, train.

### Key Limitations

- LoRA only (no full fine-tuning)
- Text environments only (but our env is already text-based)
- Cloud-dependent (API key, pay-per-token)
- Adam only optimizer
- No offline/local training option

---

## OpenReward (General Reasoning)

**URL:** https://docs.openreward.ai/

### What It Is

OpenReward is a **managed infrastructure platform** for hosting RL environments where LLM agents are trained and evaluated. Built on the **Open Reward Standard (ORS)** - an open protocol extending Anthropic's MCP with RL-specific primitives: episodes, reward signals, task splits, curriculum management.

At launch: **330+ environments**, **4.5M+ tasks**, from board games to code benchmarks to real-world simulations.

### Architecture

Two infrastructure layers:

1. **Environments:** Long-running FastAPI servers (ORS servers) exposing tasks, tools, and rewards. OpenReward hosts and autoscales these.
2. **Sandboxes:** Isolated execution containers for compute-heavy tasks. Providers: OpenReward native, E2B, Daytona, Modal.

### Agent Interaction Model

Agents interact via **tool-calling** (function-calling), not raw game inputs:

```python
from openreward import OpenReward

client = OpenReward(api_key="...")
env = client.environments.get("org/env-name")
session = env.start_session(task_id="...")

# Agent loop
while not done:
    # LLM decides which tool to call
    result = session.call_tool("submit_move", {"action": "ParryHigh"})
    reward = result.reward
    done = result.finished
```

### Custom Environment Creation

First-class support for custom environments:

```python
from openreward.environments import Environment, Server, tool, ToolOutput, TextBlock

class YomiEnv(Environment):
    @tool
    async def select_action(self, params):
        # Bridge to Godot game
        return ToolOutput(
            blocks=[TextBlock(text="...")],
            reward=hp_delta,
            finished=match_over
        )
    
    async def get_prompt(self):
        return [TextBlock(text="Game state...")]
    
    @classmethod
    def list_tasks(cls, split):
        return [{"task_id": "match_0", ...}]
```

Deploy via GitHub repo connection - OpenReward builds and hosts automatically.

### Compute Options

| Machine Size | Description |
|---|---|
| `"0.5:1"` | CPU only |
| `"nvidia-l4"` | GPU sandbox |
| Custom Docker images | Full control |

### Training Framework Integrations

- **Tinker** - primary integration for actual model training
- **Miles** - RL training
- **Slime** - RL training
- Claims compatibility with "any training library"

### Steam Dependency Question

**No Steam dependency.** The platform is entirely game-agnostic. Custom environments are Python servers wrapping whatever game logic you need. The only requirements are:
- Python ORS server implementing the base class
- Docker containerization
- GitHub repo for deployment

**However**, the game binary itself would need to run inside the Docker container. YOMI Hustle is a Steam game running on Godot - packaging it in a headless Docker container for OpenReward's sandboxes is the real challenge. Options:
1. Run Godot headless in the container (requires the game binary + Godot runtime)
2. Extract/reimplement game logic in Python (enormous effort)
3. Use OpenReward only for environment hosting, run game locally

### Relevance to Yomi Hustle

**Moderate relevance, with a significant packaging hurdle.** OpenReward provides:
- Managed environment hosting (don't need to run your own infra)
- Standard protocol for RL agent training
- Integration with Tinker for actual model training

But the game needs to run inside their containers. Since YOMI Hustle is a Steam/Godot game, not a pure Python environment, you'd need to containerize the Godot runtime + game + mod. This is doable but non-trivial.

**The Steam dependency is not about OpenReward's platform requirements** (there are none) - it's about **whether you can package the game binary into a Docker container** for their sandbox infrastructure. Running it locally and just using Tinker directly for training is simpler.

### Key Limitations

- Pricing not publicly disclosed
- Designed for LLM tool-use agents (not classical RL)
- Containerizing Godot games is non-trivial
- GPU sandboxes have limited capacity and higher cost

---

## Comparison: Tinker vs OpenReward for Yomi Hustle

| Dimension | Tinker (direct) | OpenReward + Tinker |
|---|---|---|
| **What it provides** | Training compute (GPU) | Environment hosting + training compute |
| **Game packaging** | Game runs locally, you manage | Game must run in Docker container |
| **Complexity** | Write training loop, run game locally | Write ORS server, Dockerize game, deploy |
| **Scalability** | Limited by local game instances | OpenReward autoscales environments |
| **Cost** | Tinker per-token pricing | Tinker + OpenReward sandbox costs |
| **Iteration speed** | Fast (local game, remote training) | Slower (deploy cycle for env changes) |
| **Best for** | Initial development, small-scale training | Large-scale distributed training |

### Recommended Approach

**Start with Tinker directly.** The existing daemon architecture already provides the text-based env interface needed. Write a GRPO training loop that:
1. Runs matches locally via the existing Godot mod + daemon
2. Collects trajectories (observations, actions, rewards)
3. Sends training batches to Tinker

Consider OpenReward later if you need to scale to many parallel game instances for faster training. The containerization work can wait until the training loop is proven.

---

## Other Frameworks Considered

For completeness, these are alternatives that could work but weren't selected:

| Framework | Type | Pros | Cons |
|---|---|---|---|
| **CleanRL** | Single-file RL implementations | Simple, well-documented | Not LLM-focused, would need custom integration |
| **Stable Baselines 3** | General RL | Mature, Gymnasium-compatible | Not LLM-focused |
| **RLlib** | Distributed RL | Multi-agent support | Complex, not LLM-focused |
| **TRL** | LLM RL training | HuggingFace ecosystem | Need own GPU infrastructure |
| **veRL** | LLM RL training | Good GRPO support | Need own GPU infrastructure |
| **OpenRLHF** | LLM RL training | Distributed training | Need own GPU infrastructure |

Tinker's advantage over TRL/veRL/OpenRLHF: **no GPU infrastructure needed**. You pay per-token and they handle distributed training.
