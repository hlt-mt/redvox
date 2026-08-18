from transformers import AutoProcessor, AutoModelForMultimodalLM

def load_model(*args):
    model_id = "google/gemma-4-e4b-it"

    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def generate(model_processor, model_input):
    model, processor = model_processor

    if model_input["text_only"]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": model_input["prompt"]},
                ],
            }
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": model_input["sample"]},
                    {"type": "text", "text": model_input["prompt"]},
                ],
            }
        ]

    # Process input
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    # Parse output
    return processor.parse_response(response).get("content", response)