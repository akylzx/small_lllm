import os
import torch
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from dataloader.sft_dataloader import get_sft_dataloader
from model.model import Model, ModelConfig

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[RANK {rank}]  DDP initialized.")

rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp_setup(rank, world_size)

data_files = {
    "kaz": [
        "data/sft/kk_gsm8k_benchmark_2025_jule23_train.jsonl",
        "data/sft/kk_mmlu_benchmark_2025_jule23_train.jsonl",
    ],
    "rus": [
        "data/sft/ru_gsm8k_benchmark_2025_jule23_train.jsonl",
        "data/sft/ru_mmlu_benchmark_2025_jule23_train.jsonl",
    ],
    "eng": [
        "data/sft/en_gsm8k_benchmark_2025_jule23_train.jsonl",
        "data/sft/en_mmlu_benchmark_2025_jule23_train.jsonl",
    ],
}

tokenizer_path = "llm_tokenizer/spm_bpe_tokenizer_200m/tokenizer.model"
checkpoint_path = "checkpoints_new/model_step_160000.pt"
output_dir = "checkpoints_sft/"
os.makedirs(output_dir, exist_ok=True)

pt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
config = ModelConfig()
model = Model(config).to(device=rank, dtype=pt_dtype)

checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{rank}", weights_only=False)
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"[RANK {rank}] Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)

if world_size > 1:
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    raw_model = model.module
else:
    raw_model = model

optimizer = raw_model.configure_optimizers(
    weight_decay=0.01,
    learning_rate=1e-5,
    betas=(0.9, 0.95),
    device_type="cuda"
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

batch_size = 2
dataloader = get_sft_dataloader(
    data_files=data_files,
    rank=rank,
    world_size=world_size,
    tokenizer_path=tokenizer_path,
    batch_size=batch_size
)

dataloader.sampler.set_epoch(0)

log_interval = 10
num_epochs = 3
save_interval = 2000
step = 0
start_time = time.time()
total_loss = 0.0

model.train()
for epoch in range(num_epochs):
    dataloader.sampler.set_epoch(epoch)
    for batch in dataloader:
        input_ids = batch['input_ids'].to(rank)
        attention_mask = batch['attention_mask'].to(rank)
        labels = batch['labels'].to(rank)
        language_ids = batch['language_ids'].to(rank)

        optimizer.zero_grad()

        logits, loss = model(input_ids, language_ids=language_ids, targets=labels)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        if world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        
        total_loss += loss.item()
        step += 1

        if rank == 0 and step % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"[Epoch {epoch} | Step {step}] Loss: {(total_loss/log_interval):.4f}, Elapsed: {elapsed/3600:.1f}h")
            total_loss = 0.0

        if rank == 0 and step % save_interval == 0:
            torch.save({
                "model_state_dict": raw_model.state_dict(),
                "step": step,
                "epoch": epoch
            }, os.path.join(output_dir, f"checkpoint_step{step}.pt"))
            print(f"[RANK 0] Saved checkpoint at step {step}")

dist.destroy_process_group()
