import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from new_model.model.new_model import Model, ModelConfig
from new_model.dataloader.new_dataloader import get_data_loader
from torch.nn.utils import clip_grad_norm_
from torch import GradScaler, autocast
import time
import argparse

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print('ddp setup success')

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def save_checkpoint(raw_model, opt, scheduler, current_step, cfg, world_size, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        'step': current_step,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': cfg,
        'world_size': world_size,
    }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    print(f"Checkpoint saved: {filename}")

def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f">>> RANK: {rank}, WORLD_SIZE: {world_size}", flush=True)

    setup_ddp(rank=rank, world_size=world_size)
    
    data_dirs = {
        0 : '/home/srp_base_training/data/datasets--issai--Base_LLM_Datasets/snapshots/d62ba981d9ba825905753c27ee73ed5814ebb9ed',
        1 : '/home/srp_base_training/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/ru_data/russian/good/russian',
        2 : '/home/srp_base_training/data/datasets--SRP-base-model-training--dataset_04_06_2025/snapshots/e956fda493c628e494994e0f382f46784c6fbd6a/en_data',
    }
    tokenizer_path = '/home/srp_base_training/sanzhar/llm_project/llm_tokenizer/spm_bpe_tokenizer_200m/tokenizer.model'
    output_dir = 'checkpoints_no_adpters'
    batch_size = 2  
    lr = 3e-5
    log_interval = 100
    save_interval = 10000

    pt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    cfg = ModelConfig(shared_language_adapters=False)
    model = Model(cfg).to(device=rank, dtype=pt_dtype)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model
    
    num_workers = 4
    dl = get_data_loader(
        data_dirs=data_dirs,
        rank=rank,
        world_size=world_size,
        tokenizer_path=tokenizer_path,
        max_length=cfg.block_size,
        min_length=10,
        min_text_length=50,
        batch_size=batch_size,
        num_workers=num_workers
    )

    opt = raw_model.configure_optimizers(
        weight_decay=0.01, 
        learning_rate=lr, 
        betas=(0.9, 0.95), 
        device_type='cuda'
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000000)  
    
    model.train()
    
    current_step = 0
    total_loss = 0.0
    start_time = time.time()

    epochs = 3
    
    try:
        for epoch in range(epochs):
            for batch in dl:
                if batch is None:
                    continue
                
                input_ids = batch["input_ids"].to(rank)
                labels = batch["labels"].to(rank)
                
                opt.zero_grad()
                
                logits, loss = model(input_ids, targets=labels)
                
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                opt.step()
                scheduler.step()
                
                if world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                
                total_loss += loss.item()
                current_step += 1
                
                if current_step % log_interval == 0 and rank == 0:
                    avg_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    tokens_per_sec = (current_step * batch_size * world_size * cfg.block_size) / elapsed
                    
                    print(f'Step {current_step} | '
                        f'Loss: {avg_loss:.4f} | '
                        f'LR: {scheduler.get_last_lr()[0]:.6f} | '
                        f'Tokens/s: {tokens_per_sec:.0f} | '
                        f'Elapsed: {elapsed/3600:.1f}h')
                    total_loss = 0.0
                    
                if current_step % save_interval == 0 and rank == 0:
                    print("Saving checkpoint...")
                    save_checkpoint(
                        raw_model, opt, scheduler, current_step, cfg, world_size,
                        output_dir, f'model_step_{current_step}.pt'
                    )

            if rank == 0:
                print(f"Saving checkpoint at epoch {epoch + 1}...")
                save_checkpoint(
                    raw_model, opt, scheduler, current_step, cfg, world_size,
                    output_dir, f'model_epoch_{epoch + 1}.pt'
                )
    
    except KeyboardInterrupt:
        print('Training interrupted by user.')
        if rank == 0:
            print("Saving final model state...")
            save_checkpoint(
                raw_model, opt, scheduler, current_step, cfg, world_size,
                output_dir, 'model_final.pt'
            )
        
    cleanup_ddp()

if __name__ == "__main__":
    main()