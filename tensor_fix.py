"""
Comprehensive tensor fix for Stable Diffusion WebUI
- Fixes CUDA tensor storage compatibility issues
- Fixes tensor device mismatches (Half/Float tensors)
- Handles tensor type inconsistencies
"""
import torch
import re
import gc
import sys

print("🔄 Applying comprehensive tensor compatibility fixes...")

# Fix 1: CUDA Tensor Storage compatibility fix
try:
    print("Applying CUDA Tensor Storage fix...")
    # For newer versions
    if hasattr(torch.storage, '_TypedStorage'):
        storage_class = torch.storage._TypedStorage
    # For older versions
    else:
        storage_class = torch.storage.TypedStorage

    # Create backward compatibility
    torch.Tensor.type = lambda self, *args, **kwargs: self.to(*args, **kwargs)

    # Fix for 'CUDATensorStorage'
    re_pattern = re.compile(r'CUDATensorStorage')

    original_init = storage_class.__init__
    def new_init(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], str):
            args = list(args)
            args[0] = re_pattern.sub('UntypedStorage', args[0])
            args = tuple(args)
        original_init(self, *args, **kwargs)

    storage_class.__init__ = new_init
    print('✅ CUDA Tensor Storage compatibility fix applied')

except Exception as e:
    print(f'⚠️ Error applying CUDA Tensor Storage fix: {e}')

# Fix 2: Tensor device mismatch fix
try:
    print("Applying tensor device mismatch fix...")
    
    # Wait until modules are loaded
    def apply_device_fix():
        try:
            from ldm.modules.encoders.modules import FrozenCLIPEmbedder
            
            # Store the original forward method
            original_forward = FrozenCLIPEmbedder.forward
            
            # Override the forward method to ensure tensors are on the same device
            def forward_with_device_fix(self, text):
                # Force the device to be consistent (using the model's device)
                device = next(self.transformer.parameters()).device
                
                # Call the original forward method
                result = original_forward(self, text)
                
                # Ensure the result is on the correct device
                if isinstance(result, torch.Tensor) and result.device != device:
                    result = result.to(device)
                elif isinstance(result, tuple):
                    result = tuple(x.to(device) if isinstance(x, torch.Tensor) and x.device != device else x for x in result)
                elif isinstance(result, list):
                    result = [x.to(device) if isinstance(x, torch.Tensor) and x.device != device else x for x in result]
                
                return result
            
            # Apply the patch
            FrozenCLIPEmbedder.forward = forward_with_device_fix
            print("✅ Tensor device mismatch fix applied")
            return True
        except ImportError:
            # Module not loaded yet, will try again later
            return False
        except Exception as e:
            print(f"⚠️ Error applying tensor device mismatch fix: {e}")
            return True  # Consider it applied to avoid repeated attempts
    
    # Hook into the UI initialization process
    try:
        import modules.script_callbacks as script_callbacks
        script_callbacks.on_model_loaded(lambda sd, clip: apply_device_fix())
        print("⏳ Device fix registered to apply when model loads")
    except ImportError:
        # Try immediate application if callback not available
        apply_device_fix()

except Exception as e:
    print(f"⚠️ Error setting up tensor device fix: {e}")

# Fix 3: Additional tensor type inconsistency handler
try:
    print("Applying tensor type consistency patch...")
    
    # Monkey patch torch.nn.functional to handle half/float mismatches
    if hasattr(torch, 'nn') and hasattr(torch.nn, 'functional'):
        import torch.nn.functional as F
        
        # Store original functions that might need tensor type consistency
        original_linear = F.linear
        
        # Override with type consistency enforced
        def linear_with_type_fix(input, weight, bias=None):
            if input.dtype != weight.dtype:
                weight = weight.to(input.dtype)
                if bias is not None:
                    bias = bias.to(input.dtype)
            return original_linear(input, weight, bias)
        
        # Apply the patch
        F.linear = linear_with_type_fix
        print("✅ Tensor type consistency patch applied")
    else:
        print("⚠️ Could not locate torch.nn.functional for patching")

except Exception as e:
    print(f"⚠️ Error applying tensor type consistency patch: {e}")

# Clean up memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("✅ All tensor compatibility fixes have been applied")
