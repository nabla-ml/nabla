import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import re
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURATION & PALETTE ---
color_map = {
    'N': 'black', 
    'K': 'black', 
    'Y': 'rgb(204, 108, 230)',  
    'O': 'rgb(0, 75, 173)',  
    'B':  'rgb(0, 151, 178)'   
}

# Target Palette (Purple, Teal, Dark Blue)
target_color_map = {
    'Y': 'rgb(204, 108, 230)',  
    'O': 'rgb(0, 75, 173)',  
    'B':  'rgb(0, 151, 178)'   
}

canvas_str = """
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) Y(K,Y,K) Y(K,K,K) Y(K,Y,K) Y(K,K,K) Y(K,Y,K) Y(K,K,K) Y(K,Y,K) Y(K,K,K) Y(K,Y,K) Y(K,K,K) O(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) B(K,B,K) B(K,K,K) B(K,B,K) B(K,K,K) B(K,B,K) B(K,K,K) B(K,K,K) Y(K,Y,K) Y(K,K,K) O(K,K,K) O(K,K,O) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) B(K,B,K) B(K,K,K) O(K,K,K) O(K,K,O) N(K,K,K) Y(K,Y,K) Y(K,K,K) O(K,K,K) O(K,K,O) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) B(K,B,K) B(K,K,K) O(K,K,K) O(K,K,O) Y(K,K,K) O(K,K,O) O(K,K,O) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) B(K,B,K) B(K,K,K) O(K,K,K) O(K,K,O) O(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) B(K,B,K) B(K,K,K) O(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K) N(K,K,K)
"""

# --- 2. ENGINE PARAMETERS ---
LINE_WIDTH = 3

def convert_color(color_str):
    if isinstance(color_str, str) and color_str.startswith('rgb('):
        try:
            parts = re.findall(r'\d+', color_str)
            if len(parts) == 3:
                return tuple(int(x) / 255.0 for x in parts)
        except Exception:
            pass
    return color_str

def parse_cell(cell_str):
    match = re.match(r"([A-Z])\(([^)]+)\)", cell_str)
    if not match: 
        return 'N', convert_color(color_map['N']), [convert_color(color_map['K'])] * 3, True
    
    face_key = match.group(1)
    face_color = convert_color(color_map.get(face_key, color_map['N']))
    raw_edges = [color_map.get(c.strip(), c.strip()) for c in match.group(2).split(',')]
    edge_colors = [convert_color(c) for c in raw_edges]
    
    is_N_face = (face_key == 'N')
    return face_key, face_color, edge_colors, is_N_face

lines = [line.split() for line in canvas_str.strip().split('\n')]
num_rows, num_cols = len(lines), len(lines[0])
G = nx.Graph()
face_lookup = []
h = np.sqrt(3) / 2
main_shape_nodes = set()

for r in range(num_rows):
    for c in range(num_cols):
        f_key, f_color, e_colors, is_N = parse_cell(lines[r][c])
        x_off, y_pos = c * 0.5, (num_rows - r) * h
        v_raw = [(x_off, y_pos), (x_off + 1, y_pos), (x_off + 0.5, y_pos + h)] if (r + c) % 2 == 0 else [(x_off, y_pos + h), (x_off + 1, y_pos + h), (x_off + 0.5, y_pos)]
        v_keys = [tuple(np.round(p, 4)) for p in v_raw]
        
        face_lookup.append({'nodes': v_keys, 'color': f_color, 'is_N': is_N, 'key': f_key})
        
        if f_key in ['Y', 'O', 'B']:
            for n in v_keys:
                main_shape_nodes.add(n)
        
        edges = [(v_keys[0], v_keys[2]), (v_keys[1], v_keys[2]), (v_keys[0], v_keys[1])]
        for i, (u, v_node) in enumerate(edges):
            if G.has_edge(u, v_node):
                if e_colors[i] != 'black': G[u][v_node]['color'] = e_colors[i]
            else: G.add_edge(u, v_node, color=e_colors[i])

pos_orig = {node: np.array(node) for node in G.nodes()}

# --- 3. ANIMATION PARAMETERS ---
RANDOM_SPREAD = 0.065
BREATHING_CYCLE = 2.3
FPS = 60

# NEW: Factor to reduce the spread of the Main Shape relative to background
MAIN_SHAPE_SPREAD_RATIO = 0.26

# Background Color Params
COLOR_TRANSITION_DURATION = 3.6
FRAMES_COLOR_BASE = int(COLOR_TRANSITION_DURATION * FPS)
N_COLOR_START = np.array([0.13, 0.13, 0.13])
N_COLOR_END = np.array([0.28, 0.28, 0.28])

# Precompute Colors
def get_color_array_map(source_map):
    return {k: np.array(convert_color(v)) for k, v in source_map.items() if k in ['Y', 'O', 'B']}

main_colors_start = get_color_array_map(color_map)
main_colors_target = get_color_array_map(target_color_map)

all_pos = np.array(list(pos_orig.values()))
x_range = all_pos[:,0].max() - all_pos[:,0].min()
y_range = all_pos[:,1].max() - all_pos[:,1].min()

def ease_in_out_cubic(t):
    if t < 0.5: return 4 * t * t * t
    else: return 1 - pow(-2 * t + 2, 3) / 2

def generate_node_motion():
    motion = {}
    for node in pos_orig:
        motion[node] = {
            'xf1': np.random.uniform(0.5, 1.5), 'xp1': np.random.uniform(0, 2*np.pi),
            'xf2': np.random.uniform(2.0, 3.5), 'xp2': np.random.uniform(0, 2*np.pi),
            'yf1': np.random.uniform(0.5, 1.5), 'yp1': np.random.uniform(0, 2*np.pi),
            'yf2': np.random.uniform(2.0, 3.5), 'yp2': np.random.uniform(0, 2*np.pi),
        }
    return motion

np.random.seed(42)
node_motion = generate_node_motion()

# --- 4. LOCAL RENDERING SETUP ---
fig, ax = plt.subplots(figsize=(14, 8), facecolor='black')
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(all_pos[:,0].min() - 0.5, all_pos[:,0].max() + 0.5)
ax.set_ylim(all_pos[:,1].min() - 0.5, all_pos[:,1].max() + 0.5)

poly_patches = []
n_face_patches = []
n_face_params = []
main_shape_patches = [] 

for face in face_lookup:
    p = patches.Polygon([pos_orig[n] for n in face['nodes']], facecolor=face['color'], zorder=1)
    ax.add_patch(p)
    poly_patches.append((p, face['nodes']))
    
    if face['is_N']:
        n_face_patches.append(p)
        speed = np.random.uniform(0.8, 2.5)
        offset = np.random.uniform(0, 2000)
        n_face_params.append({'speed': speed, 'offset': offset})
    elif face['key'] in ['Y', 'O', 'B']:
        main_shape_patches.append((p, face['key']))

edge_artists = []
for u, v, data in G.edges(data=True):
    z = 3 if data['color'] == 'black' else 0
    line, = ax.plot([], [], color=data['color'], linewidth=LINE_WIDTH, zorder=z, solid_capstyle='projecting')
    edge_artists.append((line, u, v))

node_scatter = ax.scatter([], [], color='black', s=LINE_WIDTH**2 * 0.6, zorder=4)

# --- 5. THE ANIMATION LOOP ---
def update(frame):
    t = frame / FPS

    # 1. Background Gray Breathing
    if FRAMES_COLOR_BASE > 0:
        base_cycle = 2 * FRAMES_COLOR_BASE
        for patch, params in zip(n_face_patches, n_face_params):
            local_frame = frame * params['speed'] + params['offset']
            frame_in_cycle = local_frame % base_cycle
            t_linear = frame_in_cycle / FRAMES_COLOR_BASE
            if t_linear <= 1.0: progress = ease_in_out_cubic(t_linear)
            else: progress = ease_in_out_cubic(2.0 - t_linear)
            color = N_COLOR_START + progress * (N_COLOR_END - N_COLOR_START)
            patch.set_facecolor(color)

    # 2. Main Shape & Color Logic
    cycle_phase = (t % BREATHING_CYCLE) / BREATHING_CYCLE
    chaos_main = (1 - np.cos(2 * np.pi * cycle_phase)) / 2
    chaos_bg = 1.0

    color_progress = (1 - np.cos(np.pi * t / BREATHING_CYCLE)) / 2

    for patch, key in main_shape_patches:
        c_start = main_colors_start[key]
        c_target = main_colors_target[key]
        curr_color = c_start * (1 - color_progress) + c_target * color_progress
        patch.set_facecolor(curr_color)

    # 3. Spatial Animation
    current_pos = {}
    scale_x_base = x_range * RANDOM_SPREAD * 0.5 
    scale_y_base = y_range * RANDOM_SPREAD * 0.5 
    
    for node, p in node_motion.items():
        wx = 0.7 * np.sin(p['xf1']*t + p['xp1']) + 0.3 * np.sin(p['xf2']*t + p['xp2'])
        wy = 0.7 * np.sin(p['yf1']*t + p['yp1']) + 0.3 * np.sin(p['yf2']*t + p['yp2'])
        
        # Apply Logic: Main shape moves significantly less than background
        if node in main_shape_nodes:
            factor = chaos_main * MAIN_SHAPE_SPREAD_RATIO
        else:
            factor = chaos_bg
            
        d_vec = np.array([wx * scale_x_base * factor, wy * scale_y_base * factor])
        current_pos[node] = pos_orig[node] + d_vec
        
    for patch, node_keys in poly_patches:
        patch.set_xy([current_pos[n] for n in node_keys])
        
    for line, u, v in edge_artists:
        p1, p2 = current_pos[u], current_pos[v]
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        
    node_scatter.set_offsets(np.array([current_pos[n] for n in G.nodes()]))
    
    return [p[0] for p in poly_patches] + [e[0] for e in edge_artists] + [node_scatter]

ani = FuncAnimation(fig, update, frames=None, interval=1000/FPS, blit=True, cache_frame_data=False)
plt.tight_layout()
plt.show()