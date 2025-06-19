import time
import json

import boto3
import botocore
from sagemaker.huggingface import HuggingFacePredictor
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

from src.llm.base import BaseLLM


class SagemakerEndpoint(BaseLLM):
    def __init__(
        self,
        endpoint,
        temperature=0.01,
        top_p=0.6,
        max_new_tokens=256,
        stop=["\n\n"]):

        super().__init__()
        self.endpoint = endpoint
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.stop = stop

    def _init_endpoint(self):
        # Config the endpoint
        self.sm_client = boto3.client("sagemaker")
        response = self.sm_client.list_endpoints(NameContains=self.endpoint)
        endpoint_names = [ep["EndpointName"] for ep in response["Endpoints"]]
        if not self.endpoint in endpoint_names:
            # create the endpoint if not exists
            create_endpoint_response = self.sm_client.create_endpoint(
                EndpointName=f"{self.endpoint}", EndpointConfigName=self.endpoint
            )
            print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")

            response = self.sm_client.list_endpoints(NameContains=self.endpoint)
            endpoint_names = [ep["EndpointName"] for ep in response["Endpoints"]]

        # check the status until in service
        start_time = time.time()
        resp = self.sm_client.describe_endpoint(EndpointName=self.endpoint)
        status = resp["EndpointStatus"]
        print("Status: " + status + "  |  time elapsed: " + "%.2f" %(time.time() - start_time) + "s", end="\r")

        while status == "Creating":
            time.sleep(5)
            resp = self.sm_client.describe_endpoint(EndpointName=self.endpoint)
            status = resp["EndpointStatus"]
            print("Status: " + status + "  |  time elapsed: " + "%.2f" %(time.time() - start_time) + "s", end="\r")

        print("Arn: " + resp["EndpointArn"])
        print("Status: " + status + "  |  time elapsed: " + str(time.time() - start_time) + "s")
        print("Delete the endpoint manually after use!!")


class TGIEndpoint(SagemakerEndpoint):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        self.model = HuggingFacePredictor(self.endpoint)

    def predict(self, prompt: str) -> str:
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_new_tokens,
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                    "stop": self.stop
                }
            }
            response = self.model.predict(payload)
            return response[0]["generated_text"][len(prompt):]
        except botocore.exceptions.ClientError as err:
            if "Input validation error" in err.response["Error"]["Message"]:
                print(err.response["Error"]["Message"])
                print(f"Prompt exceeds maximum token limit!")
                return self.predict(prompt[100:])
            elif "CUDA out of memory" in err.response["Error"]["Message"]:
                print(err.response["Error"]["Message"])
                print(f"CUDA out of memory due to parallel inference! Retrying now...")
                return self.predict(prompt)
            else:
                print("Unkown error!")
                raise err
            

class Llama2Endpoint(SagemakerEndpoint):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)

        self.endpoint = Predictor(self.endpoint)
        self.endpoint.serializer = JSONSerializer()
        self.endpoint.content_type = "application/json"

    def predict(self, prompt: str) -> str:
        try:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_new_tokens,
                    "top_p": self.top_p,
                    "temperature": self.temperature,  # Llama does not support 'stop' parameter yet
                }
            }
            response = self.endpoint.predict(payload, custom_attributes="accept_eula=true")
            return response[0]["generation"].split("\n")[0]
        except botocore.exceptions.ClientError as err:
            if "Input validation error" in err.response["Error"]["Message"]:
                print(err.response["Error"]["Message"])
                print(f"Prompt exceeds maximum token limit!")
                return self.predict(prompt[100:])
            elif "CUDA out of memory" in err.response["Error"]["Message"]:
                print(err.response["Error"]["Message"])
                print(f"CUDA out of memory due to parallel inference! Retrying now...")
                return self.predict(prompt)
            else:
                print("Unkown error!")
                raise err
            

class JumpstartEndpoint(SagemakerEndpoint):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        self.model = HuggingFacePredictor(self.endpoint)

    def predict(self, prompt: str) -> str:
        try:
            payload = {
                "text_inputs": prompt,   
                "max_length": self.max_new_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            }
            response = self.endpoint.predict(payload)
            return response['generated_texts'][0]
        except botocore.exceptions.ClientError as err:
            if "Input validation error" in err.response["Error"]["Message"]:
                print(err.response["Error"]["Message"])
                print(f"Prompt exceeds maximum token limit!")
                return self.predict(prompt[100:])
            elif "CUDA out of memory" in err.response["Error"]["Message"]:
                print(err.response["Error"]["Message"])
                print(f"CUDA out of memory due to parallel inference! Retrying now...")
                return self.predict(prompt)
            else:
                print("Unkown error!")
                raise err

class SagemakerEndpoint(BaseLLM):
    def __init__(self, endpoint_name, endpoint_config, parameters={}):
        super().__init__()
        self.parameters = parameters
        self.endpoint_name = endpoint_name

        # Config the endpoint
        self.sm_client = boto3.client("sagemaker")
        response = self.sm_client.list_endpoints(NameContains=endpoint_name)
        endpoint_names = [ep["EndpointName"] for ep in response["Endpoints"]]
        if not endpoint_name in endpoint_names:
            # create the endpoint if not exists
            create_endpoint_response = self.sm_client.create_endpoint(
                EndpointName=f"{endpoint_name}", EndpointConfigName=endpoint_config
            )
            print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")

            response = self.sm_client.list_endpoints(NameContains=endpoint_name)
            endpoint_names = [ep["EndpointName"] for ep in response["Endpoints"]]

        # check the status until in service
        start_time = time.time()
        resp = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        print("Status: " + status + "  |  time elapsed: " + "%.2f" %(time.time() - start_time) + "s", end="\r")

        while status == "Creating":
            time.sleep(5)
            resp = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            status = resp["EndpointStatus"]
            print("Status: " + status + "  |  time elapsed: " + "%.2f" %(time.time() - start_time) + "s", end="\r")

        print("Arn: " + resp["EndpointArn"])
        print("Status: " + status + "  |  time elapsed: " + str(time.time() - start_time) + "s")
        print("Delete the endpoint manually after use!!")

        if "llama" in endpoint_name:
            self.endpoint = Predictor(endpoint_name)
            self.endpoint.serializer = JSONSerializer()
            self.endpoint.content_type = "application/json"
        self.endpoint = HuggingFacePredictor(endpoint_name)

    def predict(self, prompt: str, parameters=None) -> str:
        try:
            # Can actually do batched inference with inputs as a list of prompts
            if parameters is None:
                parameters = self.parameters
            if 'llama-2' in self.endpoint_name:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens":128,
                        "top_p":0.6,
                        "temperature":0.01,
                    }
                }
                response = self.endpoint.predict(payload, custom_attributes="accept_eula=true")
                return response[0]["generation"].split("\n")[0]
            elif self.endpoint_name in ['open-llama-13b', 'dolly-v2-12b', 'jumpstart-dft-gpt-neox-20b']:
                payload = {
                    "text_inputs": prompt,   
                    "max_length":128,
                    "top_p":0.6,
                    "temperature": 0.2
                }
                response = self.endpoint.predict(payload)
                import ipdb; ipdb.set_trace()
                return response['generated_texts'][0]
            else:
                payload = {
                    "inputs": prompt,
                    "parameters": parameters
                }
                response = self.endpoint.predict(payload)
                return response[0]["generated_text"][len(prompt):]
        except botocore.exceptions.ClientError as err:
            # If exceed token limit, try shorten the token, otherwise raise the error
            if "Input validation error" in err.response["Error"]["Message"]:
                print(err.response["Error"]["Message"])
                print(f"Prompt exceeds maximum token limit!")
                return self.predict(prompt[100:], parameters)
            elif "CUDA out of memory" in err.response["Error"]["Message"]:
                print(err.response["Error"]["Message"])
                print(f"CUDA out of memory due to parallel inference! Retrying now...")
                return self.predict(prompt[100:], parameters)
            else:
                print("Unkown error!")
                raise err


if __name__ == "__main__":
    llm = SagemakerEndpoint(
        "falcon-40b-instruct",
    )

    llm.predict("aaa")