
import math
from nabla.core.sharding.spec import DeviceMesh, DimSpec, ShardingSpec

def debug_reconstruct(mesh, dim_specs):
    source_spec = ShardingSpec(mesh, dim_specs)
    rank = len(dim_specs)
    
    # Initialize mock shards
    # Value is just (size,)
    shard_values = []
    # Local size: 32 / 8 = 4
    local_size = 4
    for d_id in mesh.devices:
        shard_values.append((f"Shard_{d_id}", d_id))
        
    current_shard_descs = list(shard_values)
    
    current_active_axes = set()
    for dim in source_spec.dim_specs:
        current_active_axes.update(dim.axes)
    current_active_axes.update(source_spec.replicated_axes)

    print(f"DEBUG: Reconstruct Start. Rank={rank}. Shards={len(shard_values)}. Spec={source_spec.dim_specs} Active={current_active_axes} MeshAxes={source_spec.mesh.axis_names}")
    
    for d in range(rank - 1, -1, -1):
        if d >= len(source_spec.dim_specs):
            continue
        dim_spec = source_spec.dim_specs[d]
        print(f"DEBUG: Processing D={d}. Axes in dim: {dim_spec.axes}")
        for ax in reversed(dim_spec.axes):
            print(f"DEBUG:  >> Gathering Axis '{ax}' (Current Active: {current_active_axes})")
            groups = {}
            for val, device_id in current_shard_descs:
                signature = []
                for check_ax in sorted(list(current_active_axes)):
                    # print(f"DEBUG:      Inside check_ax loop: check={check_ax} vs ax={ax}")
                    if check_ax == ax:
                        continue
                    
                    c = source_spec.mesh.get_coordinate(device_id, check_ax)
                    signature.append((check_ax, c))

                key = tuple(signature)
                if key not in groups:
                    groups[key] = []

                my_coord = source_spec.mesh.get_coordinate(device_id, ax)
                # print(f"DEBUG:    Device {device_id} -> Key {key} Coord {my_coord} on {ax}")
                groups[key].append((my_coord, val, device_id))
            
            print(f"DEBUG:    Groups formed: {len(groups)} keys.")
            print(f"DEBUG:    Keys: {list(groups.keys())}")
            new_shard_descs = []
            for key, members in groups.items():
                members.sort(key=lambda x: x[0])
                unique_chunks = []
                # Deduplicate by coordinate
                seen_coords = set()
                
                # Verify contiguity logic (mocked)
                for m in members:
                    coord = m[0]
                    if coord not in seen_coords:
                        unique_chunks.append(m[1])
                        seen_coords.add(coord)
                
                print(f"DEBUG:    Group {key}: {len(members)} entries -> {len(unique_chunks)} unique chunks. Coords: {seen_coords}")

                # Merge
                if len(unique_chunks) > 1:
                    merged = f"Concat({unique_chunks})"
                    # Who is the new owner? logic says members[0][2] (lowest coord)
                    new_shard_descs.append((merged, members[0][2]))
                else:
                    new_shard_descs.append((unique_chunks[0], members[0][2]))

            current_shard_descs = new_shard_descs

    print("Final Shards:")
    for s in current_shard_descs:
        print(s)

# Setup Scenario
mesh = DeviceMesh("2x4_Mesh", (2, 4), ("x", "y"))
dim_spec = DimSpec(["x", "y"])
debug_reconstruct(mesh, [dim_spec])
