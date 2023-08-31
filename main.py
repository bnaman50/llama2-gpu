import time
from pathlib import Path
from typing import Tuple, Union, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from utils import full_path


def get_llama_model_and_tokenizer(
        model_pt: Path,
        device: str = "cuda",
        model_cls: Union[AutoModel, AutoModelForCausalLM] = AutoModel,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_pt)
    # This is required because LLAMA2 does not have PAD token
    # https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=OJXpOgBFuSrc
    tokenizer.pad_token = tokenizer.eos_token

    model = model_cls.from_pretrained(
        model_pt,
        device_map=device,
        torch_dtype=torch.float16,
    )
    print(f"Time taken to load model and tokenizer is {time.time() - start}")
    return model.eval(), tokenizer


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def llama2(
        sent: Union[str, List[str]],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        auto_model: bool = True,
        max_length: int = 1024
) -> torch.Tensor:
    start = time.time()
    device = torch.device(device)
    with torch.no_grad():
        sentence_encode = tokenizer(
            sent,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        if auto_model:
            sentence_embedding = model(**sentence_encode)
            sentence_embedding = sentence_embedding.last_hidden_state.detach().cpu()
        else:  # CausalLM
            sentence_embedding = model(**sentence_encode, output_hidden_states=True)
            sentence_embedding = sentence_embedding.hidden_states[-1].detach().cpu()
    sentence_embedding = mean_pooling(
        sentence_embedding, sentence_encode["attention_mask"].cpu()
    )
    print(
        f"Time taken to compute embeddings is {time.time() - start}, Shape: {sentence_embedding.shape}"
    )
    return sentence_embedding


def main():
    sent = [
        "I go to school",
        "Unequal length sentences, that's why I have a long sentence here",
    ]
    batch = 10
    sent_batch = sent * batch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = full_path("PATH TO SHARDED MODEL DIR")  # Provide the path to sharded model here
    model, tokenizer = get_llama_model_and_tokenizer(model_dir, device=device)

    for _ in range(10):
        sentence_embeddings = llama2(sent_batch, model, tokenizer, device=device)


if __name__ == "__main__":
    main()
    print("Done!")
