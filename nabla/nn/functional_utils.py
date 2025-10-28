
import nabla as nb
from nabla.nn.modules.base import Module
from nabla.nn.modules.containers import Sequential, ModuleList, ModuleDict

def functional_call(module, params_and_buffers, args, kwargs):
    original_params = {k: v for k, v in module.named_parameters()}
    original_buffers = {k: v for k, v in module.named_buffers()}
    
    try:
        new_params = {k: v for k, v in params_and_buffers.items() if k in original_params}
        new_buffers = {k: v for k, v in params_and_buffers.items() if k in original_buffers}
        
        for name, p in new_params.items():
            parts = name.split('.')
            mod = module
            for part in parts[:-1]:
                mod = mod._modules[part] 
            setattr(mod, parts[-1], p)

        for name, b in new_buffers.items():
            parts = name.split('.')
            mod = module
            for part in parts[:-1]:
                mod = mod._modules[part]
            setattr(mod, parts[-1], b)

        result = module.forward(*args, **kwargs)
        
    finally:
        # Restore original params and buffers
        for name, p in original_params.items():
            parts = name.split('.')
            mod = module
            for part in parts[:-1]:
                mod = mod._modules[part]
            setattr(mod, parts[-1], p)
            
        for name, b in original_buffers.items():
            parts = name.split('.')
            mod = module
            for part in parts[:-1]:
                mod = mod._modules[part]
            setattr(mod, parts[-1], b)
            
    return result
