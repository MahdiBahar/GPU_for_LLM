
from vllm import LLM, SamplingParams

def main():
    # 1. Load your locally‐cached model
    llm = LLM(
        model="/home/mahdi/.cache/vllm/microsoft_phi-4-mini-instruct",
        quantization="bitsandbytes",
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        max_num_seqs=1,
    )

    # 2. Set up sampling parameters
    params = SamplingParams(temperature=0.7, max_tokens=5)

    # 3. Run generation
    outputs = llm.generate(
        ["Hello, how are you?"],  # a list of prompts
        params
    )


    for req_out in outputs:
        for choice in req_out.outputs:
            print(choice.text)



if __name__ == "__main__":
    main()
