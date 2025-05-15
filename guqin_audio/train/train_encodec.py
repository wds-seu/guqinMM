# train_finetune_encodec_ddp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import EncodecModel
import torchaudio
import os
from glob import glob
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.amp as amp
import auraloss
import math
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import random

class AudioSegmentDataset(Dataset):
    def __init__(self, root_dir, segment_sec=30, sample_rate=24000, rank=0, world_size=1, split='train', val_ratio=0.1, seed=42):
        self.segment_length = segment_sec * sample_rate
        self.sample_rate = sample_rate
        self.segments = []
        self.total_segments_in_rank = 0
        self.split = split

        all_audio_files_full = sorted(glob(os.path.join(root_dir, "*.wav")))

        rng = random.Random(seed)
        rng.shuffle(all_audio_files_full)

        num_val_files = int(len(all_audio_files_full) * val_ratio)
        num_train_files = len(all_audio_files_full) - num_val_files

        if split == 'train':
            target_audio_files = all_audio_files_full[num_val_files:]
            if rank == 0:
                print(f"总文件数: {len(all_audio_files_full)}, 训练文件数: {len(target_audio_files)}")
        elif split == 'val':
            target_audio_files = all_audio_files_full[:num_val_files]
            if rank == 0:
                print(f"总文件数: {len(all_audio_files_full)}, 验证文件数: {len(target_audio_files)}")
        else:
            raise ValueError("split 参数必须是 'train' 或 'val'")

        if rank == 0:
            print(f"开始为 {split} 集扫描并切分 {len(target_audio_files)} 个音频文件...")

        files_per_rank = math.ceil(len(target_audio_files) / world_size)
        start_idx = rank * files_per_rank
        end_idx = min(start_idx + files_per_rank, len(target_audio_files))
        local_audio_files = target_audio_files[start_idx:end_idx]

        file_pbar = tqdm(local_audio_files, desc=f"Rank {rank} 切分 {split} 文件", disable=(rank!=0))
        num_segments_loaded = 0
        for filepath in file_pbar:
            try:
                wav, sr = torchaudio.load(filepath)
            except Exception as e:
                if rank == 0:
                    print(f"警告：无法加载文件 {filepath} ({split}集): {e}. 跳过。")
                continue

            if sr != self.sample_rate:
                try:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                except Exception as resample_e:
                     if rank == 0:
                         print(f"警告：重采样文件 {filepath} 失败: {resample_e}. 跳过。")
                     continue

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            wav = wav.squeeze(0)
            total_samples = wav.shape[0]

            if total_samples < self.segment_length:
                continue

            num_segments = total_samples // self.segment_length

            for i in range(num_segments):
                start = i * self.segment_length
                end = start + self.segment_length
                segment = wav[start:end]
                self.segments.append(segment)
                num_segments_loaded += 1

        self.total_segments_in_rank = num_segments_loaded
        if rank == 0:
             print(f"Rank {rank} 为 {split} 集加载了 {self.total_segments_in_rank} 个片段。")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_encodec_decoder(
    rank,
    world_size,
    data_path="./data",
    output_dir="./checkpoints_encodec",
    epochs=3,
    batch_size=4,
    lr=1e-4,
    accumulation_steps=8,
    use_amp=True,
    segment_sec=30,
    weight_decay=1e-2,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    gradient_clip_val=1.0,
    stft_loss_weight=0.5,
    val_ratio=0.1,
    seed=42
):
    setup(rank, world_size)
    device = rank

    writer = None
    if rank == 0:
        log_dir = os.path.join(output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard 日志将保存在: {log_dir}")
        print("\n--- 超参数 ---")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size (per GPU): {batch_size}")
        print(f"  Accumulation Steps: {accumulation_steps}")
        print(f"  Effective Batch Size: {batch_size * world_size * accumulation_steps}")
        print(f"  Learning Rate: {lr}")
        print(f"  Weight Decay: {weight_decay}")
        print(f"  LR Scheduler: {lr_scheduler_type}")
        print(f"  Warmup Steps: {warmup_steps if lr_scheduler_type == 'cosine' else 'N/A'}")
        print(f"  Gradient Clipping: {gradient_clip_val}")
        print(f"  Segment Seconds: {segment_sec}")
        print(f"  STFT Loss Weight: {stft_loss_weight}")
        print(f"  Validation Ratio: {val_ratio}")
        print(f"  Random Seed: {seed}")
        print("--------------\n")

    model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = True
    try:
        if hasattr(model.encoder, 'layers') and len(model.encoder.layers) >= 2:
             for p in model.encoder.layers[-2:].parameters():
                 p.requires_grad = True
             if rank == 0: print("✅ 解冻了 Encoder 的最后两层")
        else:
             if rank == 0: print("⚠️ 未找到 model.encoder.layers 或层数不足，未解冻 Encoder 层")
    except Exception as e:
        if rank == 0: print(f"⚠️ 解冻 Encoder 层时出错: {e}")

    unfrozen_quantizer_params = 0
    try:
        if hasattr(model.quantizer, 'layers'):
            for layer in model.quantizer.layers:
                if hasattr(layer, 'codebook') and hasattr(layer.codebook, 'embed'):
                    if isinstance(layer.codebook.embed, torch.Tensor):
                        layer.codebook.embed.requires_grad = True
                        unfrozen_quantizer_params += layer.codebook.embed.numel()
                    elif isinstance(layer.codebook.embed, (nn.ModuleList, nn.ParameterList)):
                         for p in layer.codebook.embed.parameters():
                             p.requires_grad = True
                             unfrozen_quantizer_params += p.numel()
                    elif isinstance(layer.codebook.embed, nn.Parameter):
                         layer.codebook.embed.requires_grad = True
                         unfrozen_quantizer_params += layer.codebook.embed.numel()

            if rank == 0:
                print(f"✅ 解冻了 Quantizer 的 {unfrozen_quantizer_params} 个 Codebook 参数")
        else:
             if rank == 0: print("⚠️ 未找到 model.quantizer.layers，未解冻 Quantizer Codebook 参数")
    except Exception as e:
         if rank == 0: print(f"⚠️ 解冻 Quantizer Codebook 参数时出错: {e}")

    if rank == 0:
        print("\n--- 需要训练的参数 ---")
        total_trainable_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_trainable_params += param.numel()
        print(f"总计可训练参数: {total_trainable_params}")
        print("----------------------\n")

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    train_dataset = AudioSegmentDataset(data_path, segment_sec=segment_sec, rank=rank, world_size=world_size, split='train', val_ratio=val_ratio, seed=seed)
    val_dataset = AudioSegmentDataset(data_path, segment_sec=segment_sec, rank=rank, world_size=world_size, split='val', val_ratio=val_ratio, seed=seed)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=True)

    scheduler = None
    warmup_scheduler = None
    if lr_scheduler_type != "none":
        num_train_segments_tensor = torch.tensor(len(train_dataset)).to(device)
        dist.all_reduce(num_train_segments_tensor, op=dist.ReduceOp.SUM)
        total_num_train_segments = num_train_segments_tensor.item()

        steps_per_epoch = math.ceil(total_num_train_segments / (batch_size * world_size * accumulation_steps))
        total_steps = steps_per_epoch * epochs
        if rank == 0:
            print(f"训练集总片段数: {total_num_train_segments}")
            print(f"每 Epoch 步数 (估算): {steps_per_epoch}")
            print(f"总训练步数 (估算): {total_steps}")

        if lr_scheduler_type == "cosine":
            if total_steps <= warmup_steps:
                 if rank == 0: print("警告: 总步数小于等于预热步数，禁用 Cosine Scheduler")
                 lr_scheduler_type = "none"
            else:
                t_max_cosine = max(1, total_steps - warmup_steps)
                scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_cosine)
                if warmup_steps > 0:
                    warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))
                else:
                    warmup_scheduler = None
        elif lr_scheduler_type == "step":
             step_size_lr = max(1, total_steps // 3)
             scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size_lr, gamma=0.1)

    stft_loss_fn = auraloss.freq.MultiResolutionSTFTLoss().to(device)
    l1_loss_fn = nn.L1Loss()
    scaler = amp.GradScaler(enabled=use_amp)
    if use_amp and rank == 0:
        print("✅ 使用混合精度训练 (torch.amp)")

    global_step = 0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_train_loss = 0.0
        num_train_steps = 0

        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [训练]")
        else:
            pbar = train_loader

        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            batch = batch.to(device, non_blocking=True)

            with amp.autocast(device_type='cuda', enabled=use_amp):
                with torch.no_grad():
                    output = model.module.encode(batch.unsqueeze(1), return_dict=True)
                    codes = output["audio_codes"]
                    scales = output["audio_scales"]

                generated_audio = model.module.decode(codes, scales)
                generated_audio = generated_audio.audio_values

                T_out = generated_audio.shape[-1]
                T_in = batch.shape[-1]
                min_T = min(T_out, T_in)

                batch_with_channel = batch.unsqueeze(1)

                s_loss = stft_loss_fn(generated_audio[:, :, :min_T], batch_with_channel[:, :, :min_T])
                l_loss = l1_loss_fn(generated_audio[:, :, :min_T], batch_with_channel[:, :, :min_T])
                loss = l_loss + stft_loss_weight * s_loss

                current_batch_loss = loss.item()
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                if gradient_clip_val is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        gradient_clip_val
                    )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                current_lr = optimizer.param_groups[0]['lr']
                if lr_scheduler_type != "none":
                    if lr_scheduler_type == "cosine":
                        if global_step < warmup_steps:
                            if warmup_scheduler: warmup_scheduler.step()
                        else:
                            if scheduler: scheduler.step()
                        current_lr = optimizer.param_groups[0]['lr']
                    elif lr_scheduler_type == "step":
                        pass

                if rank == 0:
                    loss_val = current_batch_loss
                    epoch_train_loss += loss_val
                    num_train_steps += 1

                    if writer:
                        writer.add_scalar('Loss/train_step', loss_val, global_step)
                        writer.add_scalar('LearningRate', current_lr, global_step)
                    pbar.set_postfix({
                        "Step Loss": f"{loss_val:.4f}",
                        "LR": f"{current_lr:.2e}"
                    })

                global_step += 1

        if rank == 0 and num_train_steps > 0:
            avg_epoch_train_loss = epoch_train_loss / num_train_steps
            if writer:
                writer.add_scalar('Loss/train_epoch_avg', avg_epoch_train_loss, epoch)
            print(f"Epoch {epoch+1} 平均训练损失: {avg_epoch_train_loss:.4f}")

        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        if rank == 0:
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [验证]", leave=False)
        else:
            val_pbar = val_loader

        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device, non_blocking=True)

                output = model.module.encode(batch.unsqueeze(1), return_dict=True)
                codes = output["audio_codes"]
                scales = output["audio_scales"]
                generated_audio = model.module.decode(codes, scales)
                generated_audio = generated_audio.audio_values

                T_out = generated_audio.shape[-1]
                T_in = batch.shape[-1]
                min_T = min(T_out, T_in)
                batch_with_channel = batch.unsqueeze(1)

                s_loss = stft_loss_fn(generated_audio[:, :, :min_T], batch_with_channel[:, :, :min_T])
                l_loss = l1_loss_fn(generated_audio[:, :, :min_T], batch_with_channel[:, :, :min_T])
                val_loss = l_loss + stft_loss_weight * s_loss

                total_val_loss += val_loss.item()
                num_val_batches += 1

        if num_val_batches > 0:
            val_loss_tensor = torch.tensor([total_val_loss, num_val_batches], dtype=torch.float64).to(device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)

            if rank == 0:
                avg_val_loss = val_loss_tensor[0].item() / val_loss_tensor[1].item()
                print(f"Epoch {epoch+1} 平均验证损失: {avg_val_loss:.4f}")
                if writer:
                    writer.add_scalar('Loss/val_epoch_avg', avg_val_loss, epoch)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(output_dir, "encodec_finetuned_best.pt")
                    torch.save(model.module.state_dict(), best_model_path)
                    print(f"✅ 新的最佳验证损失 {best_val_loss:.4f}，模型保存到 {best_model_path}")

        if lr_scheduler_type == "step" and scheduler:
            scheduler.step()

    if rank == 0:
        if writer:
            writer.close()
        final_save_path = os.path.join(output_dir, "encodec_finetuned_final_epoch_{}.pt".format(epochs))
        torch.save(model.module.state_dict(), final_save_path)
        print(f"✅ 最终模型状态字典保存到 {final_save_path}")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 0:
        print(f"检测到 {world_size} 个 GPU。")
    else:
        print("未检测到 GPU，将在 CPU 上运行（不支持 DDP）。")
        exit()

    data_path = "./data/merged_wav"
    output_dir = "./checkpoints_encodec"
    epochs = 1
    per_gpu_batch_size = 1
    learning_rate = 5e-5
    accumulation_steps = 8
    use_amp = True
    segment_sec = 10

    weight_decay = 1e-2
    lr_scheduler_type = "cosine"
    warmup_steps = 200
    gradient_clip_val = 1.0
    stft_loss_weight = 0.5

    val_ratio = 0.05
    seed = 42

    args = (
        world_size, data_path, output_dir, epochs, per_gpu_batch_size,
        learning_rate, accumulation_steps, use_amp, segment_sec,
        weight_decay, lr_scheduler_type, warmup_steps, gradient_clip_val, stft_loss_weight,
        val_ratio, seed
    )

    torch.multiprocessing.spawn(
        train_encodec_decoder,
        args=args,
        nprocs=world_size,
        join=True
    )
