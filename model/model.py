import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn import flash_attn_func


FLASH_AVAILABLE = False
@dataclass
class ModelConfig:
    block_size: int = 2048
    vocab_size: int = 75000
    n_layer: int = 36
    n_head: int = 16
    n_embd: int = 1536
    dropout: float = 0.05
    bias: bool = False
    
    n_languages: int = 3
    use_language_adapters: bool = True
    use_language_embeddings: bool = True
    adapter_reduction_factor: int = 16     
    
    use_flash_attention: bool = FLASH_AVAILABLE
    use_fused_mlp: bool = True
    use_rotary_embedding: bool = True
    gradient_checkpointing: bool = False
    
    shared_language_adapters: bool = False 
    adapter_residual_dropout: float = 0.05


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = False, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().view(1, seq_len, 1, self.dim)
            self._sin_cached = emb.sin().view(1, seq_len, 1, self.dim)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        self._update_cos_sin_cache(seq_len, q.device)
        
        cos = self._cos_cached[:, :seq_len, :, :]
        sin = self._sin_cached[:, :seq_len, :, :]
        
        return self.apply_rotary_emb(q, cos, sin), self.apply_rotary_emb(k, cos, sin)
    
    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos_split = cos[..., ::2]
        sin_split = sin[..., ::2]
        
        return torch.cat([
            x1 * cos_split - x2 * sin_split,
            x1 * sin_split + x2 * cos_split
        ], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.flash = config.use_flash_attention and FLASH_AVAILABLE
        
        if config.use_rotary_embedding:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config.block_size)
        else:
            self.rotary_emb = None
        
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                               .view(1, 1, config.block_size, config.block_size))
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
 
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(q, k)
        
        if self.flash:
            y = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=True)
        else:
            q = q.transpose(1, 2) 
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            att = (q @ k.transpose(-2, -1)) * self.scale
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            y = att @ v 
            y = y.transpose(1, 2)  
        
        y = y.contiguous().view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLUMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3)  
        
        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class StandardMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class LanguageAdapter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        adapter_dim = max(config.n_embd // config.adapter_reduction_factor, 32)
        
        self.down_proj = nn.Linear(config.n_embd, adapter_dim, bias=False)
        self.up_proj = nn.Linear(adapter_dim, config.n_embd, bias=False)
        self.activation = nn.GELU() 
        self.dropout = nn.Dropout(config.adapter_residual_dropout)
        
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)
        
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + self.scale * x


class MultilingualBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.ln_1 = RMSNorm(config.n_embd) if hasattr(config, 'use_rms_norm') and config.use_rms_norm else LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd) if hasattr(config, 'use_rms_norm') and config.use_rms_norm else LayerNorm(config.n_embd, bias=config.bias)
      
        if config.use_fused_mlp:
            self.mlp = SwiGLUMLP(config)
        else:
            self.mlp = StandardMLP(config)
        
        if config.use_language_adapters:
            if config.shared_language_adapters and layer_idx > config.n_layer // 2:
                self.adapters = None  
            else:
                self.adapters = nn.ModuleList([
                    LanguageAdapter(config) for _ in range(config.n_languages)
                ])
        else:
            self.adapters = None
    
    def forward(self, x: torch.Tensor, language_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        mlp_out = self.mlp(self.ln_2(x))
        
        if self.adapters is not None and language_ids is not None:
            adapted_mlp = torch.zeros_like(mlp_out)
            
            for lang_id in range(len(self.adapters)):
                mask = (language_ids == lang_id)
                if mask.any():
                    lang_indices = mask.nonzero(as_tuple=True)[0]
                    adapted_mlp[lang_indices] = self.adapters[lang_id](mlp_out[lang_indices])
            
            # Use original output for unknown languages
            unknown_mask = (language_ids < 0) | (language_ids >= len(self.adapters))
            if unknown_mask.any():
                adapted_mlp[unknown_mask] = mlp_out[unknown_mask]
            
            mlp_out = adapted_mlp
        x = x + mlp_out
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([MultilingualBlock(config, i) for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd) if hasattr(config, 'use_rms_norm') and config.use_rms_norm else LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        if config.use_language_embeddings:
            self.lang_embedding = nn.Embedding(config.n_languages, config.n_embd)
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight') or name.endswith('down_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(self, 
                idx: torch.Tensor, 
                language_ids: Optional[torch.Tensor] = None, 
                targets: Optional[torch.Tensor] = None,
                return_logits: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        if language_ids is not None and hasattr(self, 'lang_embedding'):
            lang_emb = self.lang_embedding(language_ids).unsqueeze(1)
            x = x + lang_emb
        
        # Transformer blocks
        for block in self.transformer.h:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, language_ids)
            else:
                x = block(x, language_ids)
        x = self.transformer.ln_f(x)
        
        if targets is not None or return_logits:
            logits = self.lm_head(x)
        else:
            logits = self.lm_head(x[:, [-1], :])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, RMSNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        for pn, p in self.named_parameters():
            if 'adapters' in pn and 'scale' in pn:
                no_decay.add(pn)

        if self.transformer.wte.weight is self.lm_head.weight:
            no_decay.discard('lm_head.weight')
            decay.discard('lm_head.weight')
            
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        assert len(inter_params) == 0, f"Parameters in both decay/no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters not categorized: {param_dict.keys() - union_params}"
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
   
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        
        return optimizer
    
    @torch.no_grad()
    def generate(self, 
                 idx: torch.Tensor, 
                 language_id: Optional[int] = None,
                 max_new_tokens: int = 256,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0) -> torch.Tensor:
        self.eval()
        
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            lang_ids = None
            if language_id is not None:
                lang_ids = torch.full((idx.size(0),), language_id, device=idx.device, dtype=torch.long)
            
            logits, _ = self(idx_cond, language_ids=lang_ids, return_logits=False)
            logits = logits[:, -1, :]
            
            if repetition_penalty != 1.0:
                for i in range(idx.size(0)):
                    for token in set(idx[i].tolist()):
                        if logits[i, token] < 0:
                            logits[i, token] *= repetition_penalty
                        else:
                            logits[i, token] /= repetition_penalty
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        
        return flops_achieved / flops_promised


LANGUAGE_MAP = {
    'kaz': 0,
    'rus': 1,    
    'eng': 2,  
    'unk': -1  
}