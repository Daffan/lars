
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.bettertransformer import BetterTransformer

from llm.base import BaseLLM

def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t

def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t

default_paramepters = dict(
    temp=0.2,
    top_p=0.95,
    max_length=2048,
    max_gen_length=128,
    pad_token_id=50256
)

class CodeGenModel(BaseLLM):
    def __init__(self, model_type: str, parameters: dict=default_paramepters):
        """ load models locally from pretraining
        args:
            model_type: [str] choices include ["16B-mono", "6B-mono"]
        """
        super().__init__()
        self.model_type = model_type
        hf_model_id = "Salesforce/codegen-" + model_type
        self.parameters = parameters

        model_hf = AutoModelForCausalLM.from_pretrained(hf_model_id, device_map="auto")
        self.model = BetterTransformer.transform(model_hf, keep_original_model=False)
        # customize tokenizer
        t = AutoTokenizer.from_pretrained(hf_model_id)
        t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
        t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
        t.padding_side = "left"
        t.pad_token = 50256
        self.tokenizer = t
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def predict(self, prompt: str, p: dict=None) -> str:
        if p is None:
            p = self.parameters
        else:
            p = self.parameters.update(p)

        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).input_ids

        input_ids_len = input_ids.shape[1]
        assert input_ids_len < 2048 + p["max_gen_length"]

        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            tokens = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=p["temp"],
                max_length=input_ids_len + p["max_gen_length"],
                top_p=p["top_p"],
                pad_token_id=p["pad_token_id"],
                use_cache=True,
            )
            text = self.tokenizer.batch_decode(tokens[:, input_ids_len:, ...])[0]

        return text
