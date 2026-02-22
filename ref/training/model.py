from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def load_model(model_name, quant = None):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token

    if quant in ["4bit", "8bit"]:

        quant_config = BitsAndBytesConfig(
            load_in_4bit = quant == "4bit",
            load_in_8bit = quant == "8bit",
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant = True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = quant_config,
            device_map = "auto"
        )

        print(f"Model {model_name} loaded with quantization:", quant)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype = torch.float16,
            device_map = "auto"
        )
        print(f"Model {model_name} loaded without quantization.")

    # model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer