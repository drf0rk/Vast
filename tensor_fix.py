import torch
from modules import scripts, shared, devices

class DeviceConsistencyHelper(scripts.Script):
    def title(self):
        return "Device Consistency Helper"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # No UI needed
        pass
    
    def before_process(self, p, *args):
        self.check_and_fix_tensors()
        
    def process(self, p, *args):
        self.check_and_fix_tensors()
        
    def postprocess(self, p, processed, *args):
        self.check_and_fix_tensors()
    
    def check_and_fix_tensors(self):
        try:
            # Get the current preferred device
            device = devices.device
            
            # Apply to loaded models
            for key in dir(shared):
                if key.startswith('sd_model') or 'unet' in key or 'vae' in key or 'clip' in key:
                    model = getattr(shared, key, None)
                    if model is not None and hasattr(model, 'to'):
                        try:
                            model.to(device)
                        except:
                            pass
                            
            # Safety check for SD model
            if hasattr(shared, 'sd_model') and shared.sd_model is not None:
                if hasattr(shared.sd_model, 'model'):
                    shared.sd_model.model.to(device)
                if hasattr(shared.sd_model, 'first_stage_model'):
                    shared.sd_model.first_stage_model.to(device)
                if hasattr(shared.sd_model, 'cond_stage_model'):
                    shared.sd_model.cond_stage_model.to(device)
            
            # Clear CUDA cache to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Device helper encountered an error: {e}")
            pass  # Don't crash the generation if our helper has issues