from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_from_disk
from string import Template
import argparse
import json
import os
import re

# ---------------------------------------------------------------------------
# Confidence instruction injected into student prompts
# ---------------------------------------------------------------------------
CONFIDENCE_FORMAT_LINE = (
    "Confidence: your confidence level (0.0-1.0) regarding the success of "
    "this action, must be in numerical format, no other words or explanation."
)
_BEGIN_ANCHOR = "\n\nBegin!\n"


def _inject_inline_confidence(prompt: str) -> str:
    """Insert CONFIDENCE_FORMAT_LINE before the 'Begin!' anchor in tool-use prompts."""
    if _BEGIN_ANCHOR in prompt:
        return prompt.replace(_BEGIN_ANCHOR, "\n" + CONFIDENCE_FORMAT_LINE + _BEGIN_ANCHOR, 1)
    if "\n\nBegin!" in prompt:
        return prompt.replace("\n\nBegin!", "\n" + CONFIDENCE_FORMAT_LINE + "\n\nBegin!", 1)
    return prompt + "\n" + CONFIDENCE_FORMAT_LINE


# ---------------------------------------------------------------------------
# Teacher context templates
# ---------------------------------------------------------------------------
# Mode "demo" (SDFT-style): the teacher sees a ground-truth demonstration.
#   Paper Appendix C.3.1
_TEACHER_TPL_DEMO = Template("""\
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

# Mode "correct_solution" (SDPO-style): the teacher sees a verified correct
# rollout from the student.  Paper Appendix C.3.2
_TEACHER_TPL_CORRECT = Template("""\
$orig_content

Correct solution:
$output_text

Correctly solve the original question.
""")


def _build_teacher_content(
    task_content: str,
    output_text: str,
    teacher_context_mode: str = "demo",
) -> str:
    """Build teacher prompt content using the specified template."""
    tpl = _TEACHER_TPL_DEMO if teacher_context_mode == "demo" else _TEACHER_TPL_CORRECT
    return tpl.substitute(orig_content=task_content, output_text=output_text)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CaOPD Training (Calibration-Aware On-Policy Distillation)")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_prompts_per_batch", type=int, default=32,
                        help="Gradient accumulation steps (number of prompts per optimizer step)")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="tooluse",
                        choices=["tooluse", "science"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Rollouts per prompt.  More rollouts -> finer P_acc granularity.")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--teacher_context_mode", type=str, default="demo",
                        choices=["demo", "correct_solution"],
                        help="Teacher context type: 'demo' (SDFT-style, Appendix C.3.1) "
                             "or 'correct_solution' (SDPO-style, Appendix C.3.2)")
    parser.add_argument("--no_empirical_calibration", action="store_true",
                        help="Disable CaOPD: run plain SDFT without P_acc replacement")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_tooluse_dataset(seed=42, teacher_context_mode="demo") -> tuple:
    """Load and prepare tool-use dataset with formatted prompts."""
    train_dir = "data/tooluse_data/train_data"
    train_dataset = load_from_disk(train_dir)

    def format_example(example):
        task_content = _inject_inline_confidence(example["prompt"])
        golden_text = "\n".join(example["golden_response"])
        if "Confidence:" not in golden_text:
            golden_text += "\nConfidence: 1.00"
        teacher_content = _build_teacher_content(
            task_content, golden_text, teacher_context_mode
        )
        ground_truth = json.dumps(example.get("golden_answer", []))
        return {
            "prompt": [{"role": "user", "content": task_content}],
            "teacher_prompt": [{"role": "user", "content": teacher_content}],
            "ground_truth": ground_truth,
        }

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_science_dataset(seed=42, teacher_context_mode="demo") -> tuple:
    """Load and prepare science (Chemistry MCQ) dataset with formatted prompts."""
    path = "data/science_data/train_data"
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

    def format_example(example):
        messages = example["messages"]
        task_content = messages[1]["content"] if len(messages) > 1 else messages[0]["content"]
        output_text = example["output_text"]
        teacher_content = _build_teacher_content(
            task_content, output_text, teacher_context_mode
        )
        ground_truth = json.dumps([{"Answer": example.get("answer", "")}])
        return {
            "prompt": example["messages"],
            "teacher_prompt": [
                messages[0],
                {"role": "user", "content": teacher_content},
            ],
            "ground_truth": ground_truth,
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed, args.teacher_context_mode)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed, args.teacher_context_mode)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    config = DistilConfig(
        seed=args.seed,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_prompts_per_batch,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        num_iterations=1,
        num_generations=args.num_generations,
        save_steps=100,
        max_grad_norm=1,
        report_to="none" if args.no_wandb else "wandb",
        output_dir=args.output_dir,
        log_completions=False,
        sync_ref_model=True,
        ref_model_sync_steps=1,
        ref_model_mixup_alpha=args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction=True,
        num_loss_tokens_to_skip=3,
        # CaOPD-specific
        use_empirical_calibration=not args.no_empirical_calibration,
    )

    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
