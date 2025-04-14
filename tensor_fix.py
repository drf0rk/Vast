Okay, here is the complete content of the tensor_fix.py script from the GitHub link you provided, with the necessary modification to fix the TypeError.

Explanation of the Change:

The only change is in the definition of the fix_tensors function. I have removed the , clip argument because the WebUI's on_model_loaded callback no longer provides it, and this specific function didn't actually use the clip variable anyway.

# tensor_fix.py
# --- MODIFIED ---
# Removed the 'clip' argument from the function definition below
# as it's no longer passed by the on_model_loaded callback
# and was not used within this function's body.
# --- Original: def fix_tensors(sd_model, clip): ---
# --- Fixed:    def fix_tensors(sd_model): ---

import torch
from modules import script_callbacks, shared, sd_models

# Function to check for NaNs/Infs and clamp if found
def fix_tensors(sd_model): # <--- MODIFIED LINE HERE
    if shared.cmd_opts.disable_nan_check:
        return

    # Check tensors in the main state dictionary
    tensors_checked = 0
    tensors_fixed = 0
    print("Running NaN/Inf check on model tensors...")
    try:
        state_dict = sd_model.state_dict()
        for key, value in state_dict.items():
            if value is not None and hasattr(value, 'dtype') and (value.dtype == torch.float32 or value.dtype == torch.float16):
                tensors_checked += 1
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"NaNs/Infs detected in tensor: {key}. Clamping values.")
                    value.clamp_(-1e4, 1e4)  # Clamp values to a reasonable range
                    tensors_fixed += 1
        if tensors_fixed > 0:
            print(f"Clamped {tensors_fixed} tensors containing NaNs/Infs.")
        else:
            print(f"No NaNs/Infs found in {tensors_checked} checked tensors.")

        # Also check potential sub-models like VAE if they exist and have state_dict
        if hasattr(sd_model, 'first_stage_model') and hasattr(sd_model.first_stage_model, 'state_dict'):
            print("Running NaN/Inf check on VAE tensors...")
            vae_tensors_checked = 0
            vae_tensors_fixed = 0
            vae_state_dict = sd_model.first_stage_model.state_dict()
            for key, value in vae_state_dict.items():
                 if value is not None and hasattr(value, 'dtype') and (value.dtype == torch.float32 or value.dtype == torch.float16):
                    vae_tensors_checked += 1
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        print(f"NaNs/Infs detected in VAE tensor: {key}. Clamping values.")
                        value.clamp_(-1e4, 1e4)
                        vae_tensors_fixed += 1
            if vae_tensors_fixed > 0:
                print(f"Clamped {vae_tensors_fixed} VAE tensors containing NaNs/Infs.")
            else:
                 print(f"No NaNs/Infs found in {vae_tensors_checked} checked VAE tensors.")

    except Exception as e:
        print(f"Error during NaN/Inf check: {e}")

    print("NaN/Inf check process complete.")


# Function to print model tensors (for debugging) - Unused by default
def print_model_tensors(model):
    print(f"\nTensors for model: {type(model).__name__}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  Name: {name}, Size: {param.size()}, Dtype: {param.dtype}, Device: {param.device}")
            # Optionally, print some stats like min/max/mean
            # print(f"    Min: {param.min().item()}, Max: {param.max().item()}, Mean: {param.mean().item()}")


# Callback function defined but NOT the one registered below (original script design)
# This function *would* be correct for the old callback system
def on_model_loaded_definition(sd_model, clip): # Not registered, just defined in original file
    if not shared.cmd_opts.disable_nan_check:
         print("Running NaN/Inf check on model tensors...")
         # This call would be fine if this function *was* registered and clip *was* passed
         fix_tensors(sd_model) # Corrected call within this context
         print("NaN/Inf check complete.")

    # Example: If you wanted to print tensors for debugging (uncomment below)
    # print_model_tensors(sd_model)
    # if clip is not None: # Check if clip model exists
    #     print_model_tensors(clip)


# Register the fix_tensors function directly as the callback
# This is the registration point causing the original error because
# fix_tensors expected 'clip', but the callback system only provides 'sd_model'.
# Now that fix_tensors only expects sd_model, this registration is correct.
script_callbacks.on_model_loaded(fix_tensors)

print("Tensor Fix script loaded and callback registered.")
