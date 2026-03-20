import os
import logging
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import Optional, Any, Union, List
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class NativeOnnxEncoder:
    def __init__(self, model_dir: str, repo_id: Optional[str] = None):
        target_repo = repo_id or model_dir
        
        if not os.path.isdir(model_dir):
            logger.info(f"Model directory '{model_dir}' not found. Downloading repo '{target_repo}'...")
            token = os.getenv("HF_TOKEN")
            model_dir = snapshot_download(
                repo_id=target_repo,
                local_dir=model_dir if os.path.isabs(model_dir) else None,
                token=token
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        onnx_files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")
        onnx_model_path = os.path.join(model_dir, onnx_files[0])
        
        logger.info(f"Initializing ONNX Runtime with model: {onnx_model_path}")
        self.session = ort.InferenceSession(
            onnx_model_path, 
            providers=["CPUExecutionProvider"]
        )
        self.dimension = self.session.get_outputs()[0].shape[-1]
        
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32, **kwargs) -> Any:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            embeddings = self.encode_batch([texts])
            return np.array(embeddings[0])
            
        embeddings = self.encode_batch(texts)
        return (np.array(emb) for emb in embeddings)

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to embeddings in batches."""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        outputs = self.session.run(None, ort_inputs)
        last_hidden_state = outputs[0] 
        
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = np.repeat(attention_mask[:, :, np.newaxis], last_hidden_state.shape[2], axis=2)
        
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask
        
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)
        
        return embeddings.tolist()
