import torch
import torchaudio
import os
# --- Use transformers library ---
from transformers import EncodecModel
import pathlib
import random

def generate_batch_random_music(
    model_ckpt_path="./checkpoints_encodec/encodec_finetuned_final.pt", # Path to your fine-tuned checkpoint
    output_dir="./results_transformers", # Changed output dir name slightly
    batch_size=5, # Generate 5 samples
    frame_len=750,  # Approx 10 seconds (75 frames/sec * 10 sec)
    n_codebooks=32, # For encodec_24khz
    codebook_size=1024,
    sample_rate=24000,
    device="cpu",
):
    # --- Create Output Directory ---
    try:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_path.resolve()}")
    except Exception as e:
        print(f"❌ Error creating or accessing output directory '{output_dir}': {e}")
        return

    # 1. Load transformers EncodecModel
    # Load the base model structure first
    model = EncodecModel.from_pretrained("facebook/encodec_24khz") # Or the specific base you used

    # Load the fine-tuned state dict
    if os.path.exists(model_ckpt_path):
        print(f"Loading fine-tuned state dict from {model_ckpt_path}...")
        try:
            state_dict = torch.load(model_ckpt_path, map_location=device)
            # No adaptation needed if checkpoint was saved from transformers model
            incompatible_keys = model.load_state_dict(state_dict, strict=False)
            print(f"Load State Dict Results:")
            if incompatible_keys.missing_keys:
                print(f"  Missing keys: {len(incompatible_keys.missing_keys)}")
            if incompatible_keys.unexpected_keys:
                 print(f"  Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
            print("✅ Successfully loaded fine-tuned weights.")
        except Exception as e:
            print(f"❌ Failed to load fine-tuned checkpoint: {e}")
            print("Using base Encodec model weights.")
    else:
         print(f"⚠️ Checkpoint not found at {model_ckpt_path}, using base Encodec model weights.")

    model = model.to(device)
    model.eval()

    # Verify n_codebooks matches the loaded model config
    if model.config.num_quantizers != n_codebooks:
        print(f"⚠️ Warning: Provided n_codebooks ({n_codebooks}) does not match model config ({model.config.num_codebooks}). Using model config value.")
        n_codebooks = model.config.num_quantizers

    # 2. Generate random audio batch_size times
    for batch_idx in range(batch_size):
        print(f"生成音频 {batch_idx+1}/{batch_size}...")

        # Create random codes tensor
        # Shape: [batch_size, n_codebooks, num_frames]
        # batch_size is 1 for single generation
        random_codes = torch.randint(
            low=0,
            high=codebook_size, # codebook_size is often 1024
            size=(1, n_codebooks, frame_len),
            dtype=torch.long,
            device=device
        )

        # Scales are optional for decoding, can pass None or a dummy tensor if needed
        # If your fine-tuning involved scales, you might need to handle them.
        audio_scales = torch.ones(1, 1, device=device)

        # Decode using transformers model's decode method
        with torch.no_grad():
            # The decode method expects audio_codes and optional audio_scales tensors
            embeddings = model.quantizer.decode(random_codes)

            scaled_embeddings = embeddings * audio_scales.unsqueeze(-1)
            audio_values = model.decoder(scaled_embeddings)
            print(f"Shape BEFORE squeeze: {audio_values.shape}") 

        # Save the generated audio
        output_filename = os.path.join(output_dir, f"generated_audio_{batch_idx+1}.wav")
        try:
            # audio_values shape is [32, 1, samples]
            print(f"Shape BEFORE processing: {audio_values.shape}")

            # --- Select first element, squeeze, ensure contiguous ---
            # Resulting shape is [samples]
            audio_1d = audio_values[0].squeeze().contiguous()

            # --- Explicitly reshape to [1, samples] before moving to CPU ---
            audio_to_save = audio_1d.unsqueeze(0).cpu()

            # --- Verify data type and final shape ---
            print(f"Shape being saved: {audio_to_save.shape}") # Should be [1, samples]
            print(f"Data type: {audio_to_save.dtype}") # Should ideally be torch.float32


            torchaudio.save(output_filename, 
                            audio_to_save, 
                            sample_rate=sample_rate,
                            format="wav",
                            encoding="PCM_S", 
                            bits_per_sample=16)
        
            # torchaudio.save(output_filename, audio_values.squeeze(0).cpu(), sample_rate=sample_rate)
            print(f"✅ 音频生成完成：{output_filename}")
        except Exception as e:
            print(f"❌ Error saving audio file '{output_filename}': {e}")
            continue

if __name__ == "__main__":
    generate_batch_random_music(
        model_ckpt_path="./checkpoints_encodec/encodec_decoder_final.pt", # Make sure this path is correct
        output_dir="./results_transformers", # Changed output dir name slightly
        batch_size=5, # Generate 5 samples
        frame_len=275,  # Approx 10 seconds (75 frames/sec * 10 sec)
        n_codebooks=32, # For encodec_24khz
        codebook_size=1024,
        sample_rate=24000
    )
