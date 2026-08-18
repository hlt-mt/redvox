from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import soundfile as sf


def load_model(*args):
    processor = AutoProcessor.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-4-multimodal-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    generation_config = GenerationConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")

    return model, processor, generation_config


def generate(model_processor_config, model_input):
    model, processor, generation_config = model_processor_config

    if model_input.get("text_only"):
        composed_prompt = f"<|user|>{model_input['prompt']}<|end|><|assistant|>"
        inputs = processor(
            text=composed_prompt, return_tensors="pt"
        ).to(model.device)
    else:
        composed_prompt = f"<|user|><|audio_1|>{model_input['prompt']}<|end|><|assistant|>"
        # Open audio file
        audio, samplerate = sf.read(model_input["sample"])
        inputs = processor(
            text=composed_prompt, audios=[(audio, samplerate)], return_tensors="pt"
        ).to(model.device)

    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        generation_config=generation_config,
        num_logits_to_keep=1
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response