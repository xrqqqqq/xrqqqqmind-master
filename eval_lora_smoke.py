import argparse
import warnings

import torch
from transformers import TextStreamer

from model.model import MokioMindConfig
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import init_model

warnings.filterwarnings("ignore")


def build_prompt(tokenizer, user_text: str) -> torch.Tensor:
    messages = [{"role": "user", "content": user_text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers without chat template support.
        prompt = (tokenizer.bos_token or "") + user_text

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser("LoRA smoke inference")
    parser.add_argument(
        "--lora_path", type=str, required=True, help="Path to LoRA .pth"
    )
    parser.add_argument(
        "--from_weight",
        type=str,
        default="none",
        help="Base model weight prefix (none means random base)",
    )
    parser.add_argument(
        "--save_dir", type=str, default="out", help="Base model weight dir"
    )
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--rank", type=int, default=8, help="LoRA rank used in training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--prompt", type=str, default="你好，请做一个简短自我介绍。")
    args = parser.parse_args()

    config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    model, tokenizer = init_model(
        config,
        from_weight=args.from_weight,
        save_dir=args.save_dir,
        device=args.device,
    )

    apply_lora(model, rank=args.rank)
    load_lora(model, args.lora_path)
    model.eval()

    inputs = build_prompt(tokenizer, args.prompt)
    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(args.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            streamer=streamer,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )


if __name__ == "__main__":
    main()
##.\.venv\Scripts\python.exe eval_lora_smoke.py --lora_path out/lora/lora_identity_512.pth --from_weight full_sft --save_dir out --prompt "你好，请介绍一下你自己" --max_new_tokens 64 --device cuda:0
