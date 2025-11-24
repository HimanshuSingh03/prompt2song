from pathlib import Path
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


class PromptEncoder:
    def __init__(self, model_dir: Path, device: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_dir, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            token_embeddings = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1)
            pooled = summed / counts
        return pooled[0].cpu().numpy()
