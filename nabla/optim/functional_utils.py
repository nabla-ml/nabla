
import nabla as nb
from nabla.nn import functional as F # For adam_step

def functional_adam_step(
    params: list[nb.Tensor],
    grads: list[nb.Tensor],
    optimizer_state: dict, # This will be a dict of dicts, keyed by param id
    hyperparams: dict, # Contains lr, betas, eps, weight_decay
) -> tuple[list[nb.Tensor], dict]:
    
    new_params = []
    new_optimizer_state = {}

    for i, p in enumerate(params):
        grad = grads[i]
        
        # Get state for this parameter
        p_id = id(p) # Use id as key for state
        state = optimizer_state.get(p_id, {}) # Get existing state or empty dict
        
        # Initialize state if empty
        if len(state) == 0:
            state['step'] = nb.tensor(0, dtype=nb.DType.int32) # Use tensor for step
            state['exp_avg'] = nb.zeros_like(p)
            state['exp_avg_sq'] = nb.zeros_like(p)

        state['step'] = state['step'] + nb.tensor(1, dtype=nb.DType.int32) # Increment step with int32 tensor

        new_p, new_exp_avg, new_exp_avg_sq = F.adam_step(
            p,
            grad,
            state['exp_avg'],
            state['exp_avg_sq'],
            state['step'],
            lr=hyperparams['lr'],
            beta1=hyperparams['betas'][0],
            beta2=hyperparams['betas'][1],
            eps=hyperparams['eps'],
            weight_decay=hyperparams['weight_decay'],
        )
        new_params.append(new_p)
        
        # Update state for this parameter
        new_optimizer_state[p_id] = {
            'step': state['step'],
            'exp_avg': new_exp_avg,
            'exp_avg_sq': new_exp_avg_sq,
        }
    
    return new_params, new_optimizer_state
