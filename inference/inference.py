import torch
from llm_project.new_model.model.new_model import ModelConfig, Model, LANGUAGE_MAP
from sentencepiece import SentencePieceProcessor

# Check if CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer_path = "/home/srp_base_training/sanzhar/llm_project/llm_tokenizer/spm_bpe_tokenizer_200m/tokenizer.model"
tokenizer = SentencePieceProcessor(model_file=tokenizer_path)

# checkpoint_path = "/home/srp_base_training/sanzhar/checkpoints_new/model_step_160000.pt"
checkpoint_path = "/home/srp_base_training/sanzhar/checkpoints_sft/checkpoint_step16000.pt"
# Always map to the device we're using (CPU or CUDA)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

config = ModelConfig()
model = Model(config=config)

language = "kaz"
language_id = LANGUAGE_MAP.get(language.lower())

sample = {
    'system': "Сіз сұрақтарға жауап бере алатын пайдалы көмекші боласыз",
    'user': """Адамдардың осы әрекеттерінің қайсысы жоғары сапалы топыраққа ең көп тәуелді?
            Берілген варианттардың қайсысы дұрыс?
            A: Жаяу серуендеу
            B: Аңшылық
            C: Көмір өндіру
            D: Егін өсіру""",
}

prompt = f"{sample['system']}\n{sample['user']}\nAssistant:"
tokens = 200
temperature = 0.3
top_k = 50
top_p = 0.92
repetition_penalty = 1.1

input = tokenizer.encode(prompt)
input_tensor = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}, "
          f"epoch {checkpoint.get('epoch', 'unknown')}")
else:
    state_dict = checkpoint
    print("Loaded checkpoint (legacy format)")

model.load_state_dict(state_dict)
model.to(device)

output = model.generate(idx=input_tensor, language_id=language_id, max_new_tokens=tokens, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty)

output_text = tokenizer.decode(output[0].tolist())

print("-" * 10, " Info ", "-" * 10)
print("Prompt: ", prompt)
print("Language: ", language)
print("Language id: ", language_id)
print("Max new tokens: ", tokens)
print("Device: ", device)
print("-" * 28)

print("-" * 10, " Generated text ", "-" * 10)
print(output_text)
print("-" * 36)