from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from src.llm.base import BaseLLM

class TransformerPipeline(BaseLLM):
    def __init__(
        self,
        model_name: str,
        temperature=0.01,
        top_p=0.6,
        max_new_tokens=256,
        stop=["\n\n"]):

        super().__init__()


        self.model_name = model_name.replace("|", "/")
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.stop = stop

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def predict(self, prompt: str) -> str:
        sequences = self.pipeline(
            prompt,
            max_length=2048,
            do_sample=True,
            top_p=self.top_p,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return sequences[0]["generated_text"].split(prompt)[-1]
