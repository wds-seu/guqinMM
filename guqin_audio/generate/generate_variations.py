import torch
import torchaudio
import os
from transformers import EncodecModel
import pathlib
import soundfile as sf
import numpy as np
import random
import math
import glob # <-- Import glob

def generate_recombined_variations(
    input_audio_path, # Path to the input WAV file
    model_ckpt_path="./checkpoints_encodec/encodec_finetuned_final.pt", # Path to fine-tuned checkpoint
    output_dir="./results_recombined",
    segment_duration_sec=15, # Duration of segments to split into AND target output duration
    num_variations=5,        # How many variations to generate PER SOURCE FILE
    chunk_size_ms=500,       # Duration of chunks to copy frames from (in milliseconds)
    sample_rate=24000,
    device="cpu", # Use CUDA if available
):
    # --- Create Output Directory ---
    try:
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Ensured output directory exists: {output_path.resolve()}")
    except Exception as e:
        print(f"❌ Error creating or accessing output directory '{output_dir}': {e}")
        return

    # --- Load Full Input Audio ---
    try:
        wav_full, sr = torchaudio.load(input_audio_path)
        print(f"\n>>> Processing Source File: {input_audio_path} <<<") # Indicate which file is being processed
        print(f"Loaded full input audio (Shape: {wav_full.shape}, SR: {sr})")

        # Preprocess full audio (Resample, Mono)
        if sr != sample_rate:
            wav_full = torchaudio.functional.resample(wav_full, sr, sample_rate); sr = sample_rate
        if wav_full.shape[0] > 1:
            wav_full = wav_full.mean(dim=0, keepdim=True) # Shape [1, total_samples]
        print(f"Full audio shape after processing: {wav_full.shape}")

    except Exception as e:
        print(f"❌ Error loading or processing input audio '{input_audio_path}': {e}")
        import traceback; traceback.print_exc(); return

    # 1. Load Model
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    if os.path.exists(model_ckpt_path):
        print(f"Loading fine-tuned state dict from {model_ckpt_path}...")
        try: state_dict = torch.load(model_ckpt_path, map_location=device); model.load_state_dict(state_dict, strict=False); print("✅ Successfully loaded fine-tuned weights.")
        except Exception as e: print(f"❌ Failed to load fine-tuned checkpoint: {e}. Using base weights.")
    else: print(f"⚠️ Checkpoint not found at {model_ckpt_path}. Using base weights.")
    model = model.to(device); model.eval()

    # --- Calculate frames per chunk ---
    # Encodec 24kHz model has a hop length of 320 samples (24000 / 75 frames/sec)
    hop_length = 320
    frames_per_second = sample_rate / hop_length
    chunk_size_frames = math.ceil((chunk_size_ms / 1000) * frames_per_second)
    print(f"Using chunk size: {chunk_size_ms} ms ≈ {chunk_size_frames} frames")

    # 2. Split audio and Encode All Segments
    all_segment_codes = []
    all_segment_scales = []
    segment_samples = segment_duration_sec * sample_rate
    total_samples = wav_full.shape[1]
    num_segments = math.ceil(total_samples / segment_samples)
    target_num_frames = -1 # Will be set by the first segment

    print(f"Splitting into ~{num_segments} segments of {segment_duration_sec}s...")

    with torch.no_grad():
        for i in range(num_segments):
            start_sample = i * segment_samples
            end_sample = min(start_sample + segment_samples, total_samples)
            segment_wav = wav_full[:, start_sample:end_sample]

            # Skip if segment is too short (might produce different number of frames)
            if segment_wav.shape[1] < segment_samples / 2 and i < num_segments -1 : # Allow last segment to be short
                 print(f"Skipping segment {i+1} as it's too short.")
                 continue

            print(f"Encoding segment {i+1}/{num_segments} (samples {start_sample}-{end_sample})...")
            input_segment = segment_wav.unsqueeze(0).to(device) # Add batch dim -> [1, 1, samples]

            encoded_output = model.encode(input_segment, bandwidth=6.0, return_dict=True)
            codes = encoded_output.audio_codes
            scales_output = encoded_output.audio_scales

            # --- Handle scales list ---
            if isinstance(scales_output, list):
                if len(scales_output) > 0 and torch.is_tensor(scales_output[0]): scales = scales_output[0]
                else: scales = torch.ones(1, 1, device=device)
            elif torch.is_tensor(scales_output): scales = scales_output
            else: scales = torch.ones(1, 1, device=device)

            # --- Reshape codes if 4D ---
            if codes.dim() == 4 and codes.shape[1] == 1: codes = codes.squeeze(1) # Shape [1, N, T]

            # --- Store results if valid ---
            if target_num_frames == -1:
                target_num_frames = codes.shape[2] # Number of frames for a 15s segment
                print(f"Target number of frames per segment set to: {target_num_frames}")

            # Only store segments that have the target number of frames
            if codes.shape[2] == target_num_frames:
                all_segment_codes.append(codes) # List of [1, N, T] tensors
                all_segment_scales.append(scales) # List of [1, 1] tensors
            else:
                 print(f"⚠️ Warning: Segment {i+1} produced {codes.shape[2]} frames (expected {target_num_frames}). Skipping.")

    num_valid_segments = len(all_segment_codes)
    if num_valid_segments == 0:
        print("❌ Error: No valid audio segments of the target duration could be encoded.")
        return
    print(f"Successfully encoded {num_valid_segments} valid segments.")

    # --- Calculate Average Scale ---
    if num_valid_segments > 0:
        avg_scale = torch.stack(all_segment_scales).mean(dim=0)
        print(f"Calculated average scale: {avg_scale.item():.4f}")
    else:
        avg_scale = torch.ones(1, 1, device=device) # Fallback

    # 3. Generate Variations by Recombining Frame Chunks
    with torch.no_grad():
        for i in range(num_variations):
            print(f"\n--- Generating Recombined Variation {i+1}/{num_variations} ---")

            # Initialize codes for the new variation (using shape from the first valid segment)
            variation_codes = torch.zeros_like(all_segment_codes[0]) # Shape [1, N, T]
            # --- Use Average Scale ---
            variation_scales = avg_scale

            print(f"Constructing new code sequence ({target_num_frames} frames) using chunks of {chunk_size_frames} frames...")
            current_frame = 0
            while current_frame < target_num_frames:
                # Randomly select a source segment for this chunk
                source_segment_idx = random.randint(0, num_valid_segments - 1)

                # Determine chunk boundaries for copying
                chunk_start = current_frame
                chunk_end = min(current_frame + chunk_size_frames, target_num_frames)
                num_frames_in_chunk = chunk_end - chunk_start

                # Copy the chunk of frames from the selected segment
                variation_codes[0, :, chunk_start:chunk_end] = all_segment_codes[source_segment_idx][0, :, chunk_start:chunk_end]

                # Move to the next chunk start
                current_frame += num_frames_in_chunk

            # 4. Decode the recombined codes
            print("Decoding recombined codes...")
            embeddings = model.quantizer.decode(variation_codes)
            scaled_embeddings = embeddings * variation_scales.unsqueeze(-1) # Use avg_scale here
            variation_audio_values = model.decoder(scaled_embeddings)
            print(f"Variation audio tensor shape: {variation_audio_values.shape}")

            # 5. Save the variation
            variation_filename = os.path.join(output_dir, f"recombined_{pathlib.Path(input_audio_path).stem}_{i+1}.wav")
            try:
                audio_to_save = variation_audio_values[0].squeeze().contiguous().cpu()
                print(f"Shape being saved: {audio_to_save.shape}")
                sf.write(variation_filename, audio_to_save.numpy(), sample_rate)
                print(f"✅ Recombined variation saved: {variation_filename}")
            except Exception as e:
                print(f"❌ Error saving variation file '{variation_filename}': {e}")
                import traceback; traceback.print_exc()

if __name__ == "__main__":
    # --- Configuration ---
    source_directory = "./data/processed_wav_75h/others/" # Directory containing source WAV files
    num_files_to_process = 5                             # How many random files to pick
    num_variations_per_file = 3                          # How many variations to generate for each selected file
    model_checkpoint = "./checkpoints_encodec/encodec_decoder_final.pt"
    output_directory = "./results_recombined_multi"      # Separate output directory
    segment_duration = 15
    chunk_duration_ms = 500
    # --- End Configuration ---

    # 1. Find all WAV files in the source directory
    all_wav_files = glob.glob(os.path.join(source_directory, "*.wav"))

    if not all_wav_files:
        print(f"❌ Error: No WAV files found in '{source_directory}'")
        exit()
    print(f"Found {len(all_wav_files)} WAV files in '{source_directory}'.")

    # 2. Randomly select files (ensure we don't ask for more than available)
    num_to_select = min(num_files_to_process, len(all_wav_files))
    if num_to_select < num_files_to_process:
        print(f"⚠️ Warning: Requested {num_files_to_process} files, but only {len(all_wav_files)} found. Processing {num_to_select}.")

    selected_files = random.sample(all_wav_files, num_to_select)
    print(f"Randomly selected {num_to_select} files to process:")
    for f in selected_files:
        print(f"  - {os.path.basename(f)}")

    # 3. Process each selected file
    for file_path in selected_files:
        generate_recombined_variations(
            input_audio_path=file_path,
            model_ckpt_path=model_checkpoint,
            output_dir=output_directory,
            segment_duration_sec=segment_duration,
            num_variations=num_variations_per_file,
            chunk_size_ms=chunk_duration_ms
            # sample_rate and device will use defaults from the function definition
        )

    print("\n--- Processing Complete ---")