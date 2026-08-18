from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


def load_model(*args):
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2")
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained("Qwen/Qwen3-Omni-30B-A3B-Instruct")
    return model, processor


def generate(model_processor, model_input):
    from qwen_omni_utils import process_mm_info
    model, processor = model_processor

    if model_input["text_only"]:
        user_conv_content = []
    else:
        user_conv_content = [{"type": "audio", "audio": model_input["sample"]}]

    user_conv_content.append({"type": "text", "text": model_input["prompt"]})

    conversation = [
        {
            "role": "user",
            "content": user_conv_content
        },
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
        use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids = model.generate(
        **inputs, thinker_return_dict_in_generate=False, return_audio=False,
        thinker_max_new_tokens=4096, use_audio_in_video=True)

    response = processor.batch_decode(
        text_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    return response