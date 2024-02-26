from ctransformers import AutoModelForCausalLM
import json
import config
collection = config.COLLECTION


def json_to_dict(path):
    with open(path) as schema_json:
        dict_text = schema_json.read()

    dict_obj = json.loads(dict_text)
    return dict_obj


if __name__ == "__main__":
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-v0.1-GGUF",
                                               model_file="mistral-7b-v0.1.Q4_K_M.gguf",
                                               model_type="mistral",
                                               gpu_layers=50)

    collection_dict = json_to_dict(collection)
    actions_dict = {"Create Order": collection_dict['item'][0]['item'][0],
                    "Create Consumer": collection_dict['item'][4]['item'][0],
                    "Find Item Catalog": collection_dict['item'][2]['item'][2]
                  }

    llm("<|prompter|>Given the following options, which option should be chosen to create a Customer: "
        "1. Create Order, 2. Create Customer, 3. Find Item Catalog</s><|assistant|>")

