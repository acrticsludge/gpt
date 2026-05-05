import torch
from torch.utils.data import Dataset
from typing import List


class TextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, max_seq_len: int = 1024):
        self.tokens = tokens
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.max_seq_len

    def __getitem__(self, idx: int) -> tuple:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1 : end + 1]
        return input_ids, target_ids


def create_dataset(texts: List[str], tokenizer, max_seq_len: int = 1024) -> TextDataset:
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)
    return TextDataset(torch.tensor(all_tokens, dtype=torch.long), max_seq_len)


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / self.warmup_steps
        if self.current_step < self.max_steps:
            import math
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr

    def step(self):
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.current_step += 1


def create_optimizer(model, config):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() <= 1 or "norm" in name.lower() or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=config.betas, eps=config.eps)


def train(model, dataset, config, device, save_dir: str = "checkpoints"):
    import os, time
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    model.train()

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, 
        drop_last=True, num_workers=0, pin_memory=True
    )

    optimizer = create_optimizer(model, config)
    scheduler = CosineWarmupScheduler(
        optimizer, config.warmup_steps, config.max_steps, 
        config.learning_rate
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    step, total_loss = 0, 0.0
    start_time = time.time()

    print(f"\nTraining: {model.get_num_params():,} params | Device: {device}")
    print(f"Effective batch: {config.batch_size * config.grad_accum_steps}\n")

    while step < config.max_steps:
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            if step >= config.max_steps:
                break

            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(input_ids, targets=target_ids)
            loss = loss / config.grad_accum_steps

            if use_amp and scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * config.grad_accum_steps

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                if use_amp and scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_amp and scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                if step % 100 == 0 or step == 1:
                    avg_loss = total_loss / 100
                    elapsed = time.time() - start_time
                    tps = step * config.batch_size * config.grad_accum_steps * config.max_seq_len / elapsed
                    print(f"Step {step:>6}/{config.max_steps} | Loss: {avg_loss:.4f} | LR: {scheduler.get_lr():.2e} | TPS: {tps:,.0f}")
                    total_loss = 0.0

                if step % 2000 == 0:
                    torch.save({
                        "step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "loss": avg_loss
                    }, f"{save_dir}/checkpoint_{step}.pt")

    print(f"\nDone! {(time.time()-start_time)/60:.1f} min")
    return model