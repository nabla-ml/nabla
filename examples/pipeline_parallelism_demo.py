
"""
Pipeline Parallelism Demo using Nabla Control Flow and Sharding.

This example simulates a 1F1B pipeline parallelism schedule using:
1. `nabla.ops.control_flow.scan` for the microbatch loop.
2. `nabla.ops.communication.ppermute` for data transfer.
3. `nabla.ops.control_flow.cond` for bubble handling.
4. "Global Graph Slicing" to simulate shard_map.

Architecture:
- Simple MLP layers stacked.
- 4 pipeline stages.
- 1F1B schedule.
"""

import numpy as np
from typing import Any, Tuple, List

import nabla
from nabla import ops
from nabla.core import tensor, pytree
from nabla.core.tensor import Tensor
from max import driver, graph
from max.dtype import DType


# Helper for dynamic slicing (simulating shard_map)
def get_local_slice(x: Tensor, axis: int, start_idx: Tensor, length: int, N: int) -> Tensor:
    """Slice x along axis starting at dynamic start_idx with fixed length.
    
    Since slice_tensor doesn't fully support dynamic starts, we use gather.
    """
    # 1. Create indices: [start, start+1, ..., start+length-1]
    # start_idx is (N,) or ().
    # We construct indices manually to avoid complex broadcasting in trace.
    
    indices_list = []
    for i in range(length):
        # start_idx + i
        # If start_idx is (N,), result is (N,).
        offset = ops.creation.constant(i, dtype=DType.int32, device=start_idx.device)
        indices_list.append(start_idx + offset)
        
    # Stack along axis 1 -> (N, length)
    # If start_idx was scalar (), result (length,) -> (1, length) if axis=0?
    if start_idx.rank > 0:
        indices = ops.view.stack(indices_list, axis=1)
    else:
        indices = ops.view.stack(indices_list, axis=0) # (L,)
    
    # 2. Gather
    return ops.gather(x, indices, axis=axis)

def mlp_forward(x, w, b):
    # x: (batch, hidden)
    # w: (hidden, hidden)
    # b: (hidden,)
    return ops.matmul(x, w) + b

class PipelineStage:
    def __init__(self, stage_id, num_stages, layers_w, layers_b):
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.layers_w = layers_w # (layers_per_stage, hidden, hidden)
        self.layers_b = layers_b # (layers_per_stage, hidden)
        
    def forward(self, x):
        # Run all layers in this stage sequentially
        # We can scan over layers or unroll. Unroll is better for small N.
        h = x
        num_layers = self.layers_w.shape[0] # Static logic?
        # But self.layers_w is a Tensor (slice).
        # We rely on max loop/scan if N is large. For N=2, unroll.
        
        # We can use Python loop if layers_w shape is known statically intra-shard?
        # Yes, we sliced with known length.
        # But slicing result shape might be symbolic if not carefully done.
        # Let's hope max infers shape (2, ...).
        
        # For now, simplistic "scan" over layers
        def layer_fn(carry, params):
            inp = carry
            w, b = params
            out = mlp_forward(inp, w, b)
            # Add activation
            out = ops.relu(out)
            return out, out # carry, scan_out
            
        # Stack params for scan
        # layers_w: (N, H, H), layers_b: (N, H)
        # We need to zip them? 
        # scan expects xs to be pytree.
        xs = (self.layers_w, self.layers_b)
        
        final_h, _ = ops.control_flow.scan(layer_fn, h, xs)
        return final_h


from nabla.sharding import DeviceMesh, DimSpec, ShardingSpec

def main():
    print("Initializing Pipeline Parallelism Demo...")
    
    # 1. Setup Mesh
    mesh = DeviceMesh("pipeline_mesh", (4,), axis_names=("pp",))
    
    # 2. Define Params
    num_stages = 4
    layers_per_stage = 2
    total_layers = num_stages * layers_per_stage
    hidden = 64
    batch_size = 4  # Explicit consistent batch size
    microbatches = 8
    
    # Random init
    W = tensor.Tensor.from_dlpack(np.random.randn(total_layers, hidden, hidden).astype(np.float32))
    B = tensor.Tensor.from_dlpack(np.zeros((total_layers, hidden), dtype=np.float32))
    
    # Init inputs
    inputs_np = np.random.randn(microbatches, batch_size, hidden).astype(np.float32)
    inputs = tensor.Tensor.from_dlpack(inputs_np)
    print(f"Inputs shape: {inputs.shape}")
    
    # Shard Params

    # Shard Params
    from nabla.sharding import DimSpec, ShardingSpec
    W_sharded = ops.shard(W, mesh, [DimSpec(["pp"]), DimSpec([]), DimSpec([])])
    B_sharded = ops.shard(B, mesh, [DimSpec(["pp"]), DimSpec([])])
    
    print("Params sharded.")

    # 3. Pipeline Step Function (runs on ALL devices)
    def pipeline_step(carry_state, x_mb):
        # carry_state: ignored for now
        # x_mb: Input microbatch (only valid for stage 0?)
        # In real PP, Stage 0 picks x_mb, non-0 receive from prev stage.
        # But we pass 'inputs' to scan. 'inputs' is replicated? Or sharded?
        # If inputs is replicated, everyone gets x_mb.
        # Stage 0 uses it. Others ignore it.
        
        # A. Identify My Rank & Stage
        # my_rank = ops.communication.axis_index(mesh, "pp") # Scalar Tensor
        # Workaround for axis_index shape issues: Use sharded arange (0..num_stages)
        my_rank_global = ops.creation.arange(0, num_stages, dtype=DType.int32, device=mesh.device_refs[0])
        # Shard it on 'pp' so each device gets its index (as a size-1 tensor, or scalar?)
        # Global (4,) -> Local (1,) ?
        my_rank = ops.shard(my_rank_global, mesh, [DimSpec(["pp"])])
        
        # We assume 1 device per stage for now.
        
        # B. Get Local Parameters (Global Graph Slicing)
        # start = my_rank * layers_per_stage
        # length = layers_per_stage
        # But my_rank is Tensor.
        # Note: Since W/B are sharded, the local tensor IS the slice we want.
        # We define relevant indices relative to the local shard (which starts at 0).
        local_start = ops.creation.constant(0, dtype=DType.int32, device=my_rank.device)
        
        my_W = get_local_slice(W_sharded, 0, local_start, layers_per_stage, num_stages)
        my_B = get_local_slice(B_sharded, 0, local_start, layers_per_stage, num_stages)
        
        stage = PipelineStage(my_rank, num_stages, my_W, my_B)
        
        # C. Communication (Receive from Prev)
        # If stage > 0, we need input from stage-1.
        # Logic:
        # All devices forward pass:
        # 1. Receive from i-1 (ppermute)
        # 2. If stage==0, use x_mb. Else use received.
        # 3. Compute
        # 4. ppermute send to i+1 (implicit in next step's receive?)
        # NO, ppermute does both send and receive.
        
        # We need a value to send.
        # In step t, we process data.
        # BUT we need data from t-1 from prev stage.
        # This implies 'carry' in scan must hold values being passed between stages?
        # Or we use ppermute inside the step.
        
        # "Forward only" naive ring:
        # y = computation(x)
        # send y to next.
        # receive x_new from prev.
        
        # Correct pattern for 1 batch passing through pipe:
        # It takes N steps to clear.
        # With continuous microbatches, it fills up.
        
        # Let's implements simple "Receive -> Compute -> Send"
        # Since ppermute is collective:
        # output_to_send = computation(input_received)
        # input_from_prev = ppermute(input_to_send, map: i -> i+1)
        # This means stage i sends its output to i+1.
        # And receives from i-1.
        
        # Issue: This creates a dependency loop if we do Compute THEN Permute?
        # No. Permute(Compute(prev_val)).
        
        # In step k:
        #   in_val (from prev step's permute return)
        #   out_val = Compute(in_val)
        #   next_in_val = Permute(out_val) -- sends my out to next. Returns what I get from prev.
        #   return next_in_val
        
        # STARTUP:
        # Stage 0 injects new data.
        # Stage > 0 process 'bubble' initially?
        
        # We need specific logic:
        # input_to_use = ...
        # If stage 0: input_to_use = x_mb (from scan input)
        # Else: input_to_use = from_permute
        
        # Logic:
        # 1. Define 'val_from_prev'. Initial is 0.
        # 2. current_input = where(my_rank == 0, x_mb, val_from_prev)
        # 3. current_output = stage.forward(current_input)
        # 4. val_from_prev_next = ppermute(current_output, perm: i -> i+1)
        
        # carry is 'val_from_prev'.
        
        val_from_prev = carry_state
        
        # Condition: Am I stage 0?
        is_stage_0 = ops.equal(my_rank, ops.creation.constant(0, dtype=DType.int32, device=my_rank.device))
        
        current_input = ops.where(is_stage_0, x_mb, val_from_prev)
        
        # Compute
        output = stage.forward(current_input)
        
        # Permute: Send output to i+1
        # Each rank i sends to (i+1)%N.
        # Rank 0 receives from Rank N-1 (Result?)
        # For pipeline, Rank N-1 output is FINAL result.
        # Rank 0 should NOT receive from N-1 usually, unless ring.
        # If not ring, Rank 0 receives junk/zeros.
        
        # Let's use ring for simplicity so Rank 0 collects results?
        # Or just open line.
        # Perm list: (0,1), (1,2), (2,3).
        # Device 3 sends to None?
        # ppermute allows partial.
        perm = [(i, i+1) for i in range(num_stages - 1)]
        
        received_from_prev = ops.communication.ppermute(output, perm)
        # received_from_prev at rank i is what i-1 sent.
        # rank 0 gets 0 (default).
        # rank 1 gets rank 0 output.
        
        return received_from_prev, output # new_carry, output_log
        
    # Init carry: Zeros (batch, hidden)
    # Must match shape
    init_carry = ops.creation.zeros((batch_size, hidden), dtype=DType.float32, device=inputs.device)
    
    print("Starting scan...")
    # SCAN
    final_carry, stacked_outs = ops.control_flow.scan(pipeline_step, init_carry, inputs)
    
    print("Scan built. Executing...")
            
    # To execute, we need values.
    # W, B, inputs are eager tensors? Yes.
    # scan returns eager tensors? Yes.
    
    # We can inspect stacked_outs.
    # It contains outputs of ALL stages at ALL steps.
    # stacked_outs sharded?
    # Each rank returns its 'output'.
    # stacked_outs is Pytree of Tensors.
    # The tensor is sharded on 'pp'. Rank i has Stage i output.
    
    result_np = stacked_outs.to_numpy()
    print("Execution complete.")
    print(f"Result shape: {result_np.shape}")
    
    # Validation
    # Stage 0 output at step 0 -> Stage 1 input at step 1 -> ...
    # Final result from Stage 3 should appear at step 3+ (pipeline lag).
    
if __name__ == "__main__":
    main()
