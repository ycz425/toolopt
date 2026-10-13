import torch
from datasets import Dataset

from app.agents.prompting import build_system_prompt, build_user_prompt, format_action
from app.data.preference import PreferencePair


def dpo_dataset(preference_pairs: list[PreferencePair]) -> Dataset:
    """Preference pairs in TRL's conversational format. The prompt is built exactly as for SFT and inference (system
    prompt with the tools, user prompt with the task and the history so far), and the responses exactly as SFT
    targets, so DPO scores the same text the model was trained to produce. TRL applies the chat template."""
    return Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": build_system_prompt(pair.state.available_tools)},
                {"role": "user", "content": build_user_prompt(pair.state.task, pair.state)},
            ],
            "chosen": [{"role": "assistant", "content": format_action(pair.chosen)}],
            "rejected": [{"role": "assistant", "content": format_action(pair.rejected)}],
        }
        for pair in preference_pairs
    ])


class _TorchLogProbFunction:
    """Plain-PyTorch stand-in for TRL's Triton `ChunkedLogProbFunction`, which TRL 1.15 always uses to score tokens but
    which is unavailable without Triton (macOS, and Trillium's torch build). Same call signature and outputs.

    It materializes the logits of the scored tokens only. DPO scores just the response tokens (the prompt is masked),
    a few dozen per batch, so the [tokens, vocab] logits are small and no chunking is needed. Gradients come from
    regular autograd.
    """

    @staticmethod
    def apply(hidden, weight, bias, targets, temperature, chunk_size, final_logit_softcapping=None, logit_scale=1.0,
              outputs=("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")):
        logits = (hidden @ weight.T).float()
        if bias is not None:
            logits = logits + bias.float()
        logits = logits * logit_scale
        if final_logit_softcapping is not None:
            logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
        logits = logits / temperature

        log_probs_all = torch.log_softmax(logits, dim=-1)
        log_probs = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():  # logging-only statistics
            probs = log_probs_all.exp()
            entropy = -(probs * log_probs_all).sum(-1) if "entropy" in outputs else None
            log_sum_sq_probs = torch.logsumexp(2 * log_probs_all, dim=-1) if "log_sum_sq_probs" in outputs else None
            mean_logits = logits.mean(-1) if "mean_logits" in outputs else None
            is_top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
        return log_probs, entropy, log_sum_sq_probs, mean_logits, is_top1


def ensure_logprob_kernel() -> None:
    """Install the PyTorch fallback when TRL's Triton log-prob kernel isn't available."""
    import trl.trainer.utils as trl_utils

    if trl_utils._ChunkedLogProbFunction is None:
        print("Triton not available: scoring tokens with the PyTorch fallback instead of TRL's fused kernel")
        trl_utils._ChunkedLogProbFunction = _TorchLogProbFunction
