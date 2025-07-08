import torch
import os
from new_model.model.new_model import ModelConfig, Model, LANGUAGE_MAP
from new_model.dataloader.new_dataloader_sinlge import get_data_loader
from sentencepiece import SentencePieceProcessor

def perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            language_ids = batch['language_ids'].to(device)

            logits, loss = model(input_ids, language_ids, targets=labels)
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        return float('inf')
    
    print('Total tokens:', total_tokens)
    avg_loss = total_loss / total_tokens
    perplexity_score = torch.exp(torch.tensor(avg_loss, device=device))
    return perplexity_score.item()

def perplexity_unreduced(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            language_ids = batch['language_ids'].to(device)

            logits, loss = model(input_ids, language_ids, targets=labels)
            
            # If loss is unreduced (per-token losses)
            if loss.dim() > 0:
                valid_mask = (labels != -100)
                valid_losses = loss[valid_mask]
                total_loss += valid_losses.sum().item()
                total_tokens += valid_mask.sum().item()
            else:
                # If loss is already reduced
                valid_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens

    if total_tokens == 0:
        return float('inf')
    
    print('Total tokens:', total_tokens)
    avg_loss = total_loss / total_tokens
    perplexity_score = torch.exp(torch.tensor(avg_loss, device=device))
    return perplexity_score.item()

device = "cuda" if torch.cuda.is_available() else "cpu"
pdtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

data_files = [
    ('/home/srp_base_training/sanzhar/llm_project/new_model/russian_eval_texts.json', 1),
    ('/home/srp_base_training/sanzhar/llm_project/new_model/kazakh_eval_texts.json', 0),
    ('/home/srp_base_training/sanzhar/llm_project/new_model/english_eval_texts.json', 2)
]

checkpoint_path = "/home/srp_base_training/sanzhar/checkpoints/model_step_43000.pt"
tokenizer_path = "/home/srp_base_training/sanzhar/llm_project/llm_tokenizer/spm_bpe_tokenizer_200m/tokenizer.model"

tokenizer = SentencePieceProcessor(model_file=tokenizer_path)
print('Loaded tokenizer')

batch_size = 1
num_workers = 0

config = ModelConfig()
model = Model(config)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
print('Loaded checkpoint')

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict=state_dict)
model.to(pdtype).to(device=device)
model.eval()

text = "once upon a"
enc = tokenizer.encode(text)
input_ids = torch.tensor(enc, dtype=torch.long, device=device).unsqueeze(0)
language_ids = torch.tensor([LANGUAGE_MAP['eng']], dtype=torch.long, device=device)
targets = input_ids[:, 1:].contiguous()
input_for_loss = input_ids[:, :-1].contiguous()

with torch.no_grad():
    logits, loss = model(input_for_loss, language_ids=language_ids, targets=targets)
    ppl = torch.exp(loss)
print(f"loss={loss.item():.4f}  |  perplexity={ppl.item():.2f}")

for path, lang_id in data_files:
    dl = get_data_loader(
        files=[(path, lang_id)], 
        rank=rank, 
        world_size=world_size, 
        tokenizer_path=tokenizer_path, 
        max_length=config.block_size, 
        min_length=5, 
        min_text_length=10, 
        batch_size=batch_size, 
        num_workers=num_workers
    )
    print('Loaded dataset')
    perplexity_score = perplexity(model, dl, device)
    print(f'Language id: {lang_id}, perplexity score: {perplexity_score:.4f}')

dl = get_data_loader(
    files=data_files, 
    rank=rank, 
    world_size=world_size, 
    tokenizer_path=tokenizer_path, 
    max_length=config.block_size, 
    min_length=0, 
    min_text_length=0, 
    batch_size=batch_size, 
    num_workers=num_workers
)
print('Loaded combined dataset')

perplexity_score = perplexity(model, dataloader=dl, device=device)
print(f'Combined perplexity score: {perplexity_score:.4f}')