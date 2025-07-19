
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
from vllm import LLM, SamplingParams

# --- Configuration ---
# Default local model path (adjust as needed)
DEFAULT_MODEL_PATH = os.getenv(
    "VLLM_MODEL_PATH",
    "/home/mahdi/.cache/vllm/microsoft_phi-4-mini-instruct"
)

# --- Pydantic models ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7

# --- Initialize FastAPI ---
app = FastAPI(title="vLLM FastAPI Service")

# --- Lazy engine loader ---
_engine_cache = {}
def get_engine():
    if DEFAULT_MODEL_PATH not in _engine_cache:
        # instantiate LLM engine once
        _engine_cache[DEFAULT_MODEL_PATH] = LLM(
            model=DEFAULT_MODEL_PATH,
            quantization="bitsandbytes",
            max_model_len=4096,
            gpu_memory_utilization=0.6,
            max_num_seqs=1,
        )
    return _engine_cache[DEFAULT_MODEL_PATH]

# --- Endpoints ---
@app.post("/v1/completions")
def completions(req: CompletionRequest):
    try:
        llm = get_engine()
        temperature = req.temperature if req.temperature is not None else 0.7
        max_tokens = req.max_tokens if req.max_tokens is not None else 128
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        out = llm.generate([req.prompt], params)
        # Single-choice
        text = out[0].outputs[0].text
        return {"choices": [{"text": text}]}  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        # concatenate messages into a single prompt
        system = next((m.content for m in req.messages if m.role == "system"), "")
        user_msgs = [m.content for m in req.messages if m.role == "user"]
        prompt = (system + "\n" if system else "") + "\n".join(user_msgs)

        llm = get_engine()
        temperature = req.temperature if req.temperature is not None else 0.7
        max_tokens = req.max_tokens if req.max_tokens is not None else 128
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        out = llm.generate([prompt], params)
        text = out[0].outputs[0].text
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Run with: uvicorn app:app --host 0.0.0.0 --port 8000 ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))