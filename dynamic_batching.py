# dynamic_batching.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os

from vllm import LLM, SamplingParams

DEFAULT_MODEL_PATH = os.getenv(
    "VLLM_MODEL_PATH",
    "/home/mahdi/.cache/vllm/microsoft_phi-4-mini-instruct"
)

app = FastAPI(title="vLLM Dynamic‑Batching Service")

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = LLM(
            model=DEFAULT_MODEL_PATH,
            quantization="bitsandbytes",
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048,
            max_num_seqs=4,
            max_model_len=4096,
            gpu_memory_utilization=0.6,
        )
    return _engine

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7

@app.post("/v1/completions")
def completions(req: CompletionRequest):
    try:
        engine = get_engine()
        params = SamplingParams(
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        out = engine.generate([req.prompt], params)
        text = out[0].outputs[0].text
        return {"choices":[{"text": text}]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
