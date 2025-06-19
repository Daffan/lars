from src.llm.base import BaseLLM
from src.llm.openai import OpenAILLM
from src.llm.transformer_pipeline import TransformerPipeline
from src.llm.sagemaker_endpoint import TGIEndpoint, JumpstartEndpoint, Llama2Endpoint

endpoint_to_class = {
    "gpt-3.5-turbo": OpenAILLM,
    "gpt-3.5-turbo-instruct": OpenAILLM,
    "gpt-3.5-turbo-16k": OpenAILLM,
    "text-davinci-003": OpenAILLM,
    "falcon-40b-instruct": TGIEndpoint,
    "tiiuae/falcon-40b-instruct": TransformerPipeline,
    "falcon-7b-instruct": TGIEndpoint,
    "vicuna-13b-1vdot3": TGIEndpoint,
    "Llama2-70b": Llama2Endpoint,
}