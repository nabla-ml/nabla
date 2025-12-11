import os
import time
import argparse
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap
import numpy as np
import imageio
from PIL import Image, ImageDraw

import warnings
import logging


# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*os.fork().*")
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# --- Pure JAX CartPole Environment ---

class PureJaxCartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masscart + self.masspole)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Thresholds
        self.theta_threshold_radians = 12 * 2 * jnp.pi / 360
        self.x_threshold = 2.4

    def reset(self, key):
        # Random initial state: uniform between -0.05 and 0.05
        state = jax.random.uniform(key, shape=(4,), minval=-0.05, maxval=0.05)
        return state

    def step(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = jax.lax.select(action == 1, self.force_mag, -self.force_mag)
        
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        next_state = jnp.array([x, x_dot, theta, theta_dot])

        # Termination conditions
        done = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )
        
        # Reward is 1.0 for every step taken
        reward = 1.0
        
        return next_state, reward, done

# --- Custom Renderer ---

def render_cartpole(state, width=600, height=400):
    """
    Renders the CartPole state using PIL.
    state: [x, x_dot, theta, theta_dot]
    """
    x, _, theta, _ = state
    
    # Scale factors
    world_width = 2.4 * 2
    scale = width / world_width
    carty = 250 # Lower the cart (was 100)
    polewidth = 10.0
    polelen = scale * 1.0
    cartwidth = 50.0
    cartheight = 30.0

    # Create image
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw track
    draw.line([(0, carty), (width, carty)], fill=(0, 0, 0), width=1)

    # Cart position
    cartx = x * scale + width / 2.0
    cart_l = cartx - cartwidth / 2
    cart_r = cartx + cartwidth / 2
    cart_t = carty - cartheight / 2
    cart_b = carty + cartheight / 2
    
    # Draw cart
    draw.rectangle([cart_l, cart_t, cart_r, cart_b], fill=(0, 0, 0))

    # Pole position
    l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
    
    # Rotate pole
    # NOTE: In CartPole, Positive Theta is Clockwise (Right).
    # Standard 2D rotation is Counter-Clockwise.
    # So we must rotate by -theta.
    rotation_angle = -theta
    
    coords = []
    for px, py in [(l, b), (l, t), (r, t), (r, b)]:
        # Rotate
        px_rot = px * np.cos(rotation_angle) - py * np.sin(rotation_angle)
        py_rot = px * np.sin(rotation_angle) + py * np.cos(rotation_angle)
        
        # Translate (flip y because PIL y is down)
        screen_x = cartx + px_rot
        screen_y = carty - py_rot 
        coords.append((screen_x, screen_y))
        
    # Draw pole
    draw.polygon(coords, fill=(204, 153, 102))

    # Draw axle
    draw.ellipse([cartx - 2, carty - 2, cartx + 2, carty + 2], fill=(127, 127, 255))

    return np.array(img)

# --- Neural Networks ---

def orthogonal_init(key, shape, scale=1.0):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = jax.random.normal(key, flat_shape)
    u, _, vt = jnp.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else vt
    q = q.reshape(shape)
    return scale * q

def init_mlp_layer(key, in_dim, out_dim, scale=1.0, bias_init=0.0):
    k_w, k_b = jax.random.split(key)
    return {
        'w': orthogonal_init(k_w, (in_dim, out_dim), scale),
        'b': jnp.full((out_dim,), bias_init)
    }

def init_actor_critic(key, obs_dim, action_dim, hidden_dim=64):
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    return {
        'actor': {
            'l1': init_mlp_layer(k1, obs_dim, hidden_dim, scale=np.sqrt(2)),
            'l2': init_mlp_layer(k2, hidden_dim, hidden_dim, scale=np.sqrt(2)),
            'head': init_mlp_layer(k3, hidden_dim, action_dim, scale=0.01)
        },
        'critic': {
            'l1': init_mlp_layer(k4, obs_dim, hidden_dim, scale=np.sqrt(2)),
            'l2': init_mlp_layer(k5, hidden_dim, hidden_dim, scale=np.sqrt(2)),
            'head': init_mlp_layer(k6, hidden_dim, 1, scale=1.0)
        }
    }

def forward_mlp(params, x, activation=jax.nn.tanh):
    x = x @ params['l1']['w'] + params['l1']['b']
    x = activation(x)
    x = x @ params['l2']['w'] + params['l2']['b']
    x = activation(x)
    x = x @ params['head']['w'] + params['head']['b']
    return x

def get_action_logits(actor_params, obs):
    return forward_mlp(actor_params, obs)

def get_value(critic_params, obs):
    return forward_mlp(critic_params, obs).squeeze(-1)

# --- Optimizer ---

def init_adam_state(params):
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return {'m': m, 'v': v, 'step': 0}

def adam_update(grads, opt_state, params, lr, max_grad_norm=None, beta1=0.9, beta2=0.999, eps=1e-8):
    step = opt_state['step'] + 1
    m = opt_state['m']
    v = opt_state['v']
    
    if max_grad_norm is not None:
        leaves, _ = jax.tree_util.tree_flatten(grads)
        total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
        clip_coef = jnp.minimum(max_grad_norm / (total_norm + 1e-6), 1.0)
        grads = jax.tree.map(lambda g: g * clip_coef, grads)
    
    m = jax.tree.map(lambda m_i, g_i: beta1 * m_i + (1 - beta1) * g_i, m, grads)
    v = jax.tree.map(lambda v_i, g_i: beta2 * v_i + (1 - beta2) * (g_i ** 2), v, grads)
    
    m_hat = jax.tree.map(lambda m_i: m_i / (1 - beta1 ** step), m)
    v_hat = jax.tree.map(lambda v_i: v_i / (1 - beta2 ** step), v)
    
    params = jax.tree.map(lambda p_i, m_h, v_h: p_i - lr * m_h / (jnp.sqrt(v_h) + eps), params, m_hat, v_hat)
    return params, {'m': m, 'v': v, 'step': step}

# --- Training Logic ---

def train(args):
    run_name = f"PureJaxCartPole_{args.seed}_{int(time.time())}"
    print(f"Running JAX-Native PPO: {run_name}")
    
    # Hyperparameters
    num_envs = args.num_envs
    num_steps = args.num_steps
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // args.num_minibatches
    
    # Calculate total iterations
    total_iterations = args.total_timesteps // batch_size
    if total_iterations == 0:
        print("Warning: total_timesteps < batch_size. Setting iterations to 1.")
        total_iterations = 1
        
    checkpoint_freq = args.checkpoint
    
    # Initialize Environment
    env = PureJaxCartPole()
    
    # Initialize State
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    
    params = init_actor_critic(init_key, obs_dim=4, action_dim=2, hidden_dim=64)
    opt_state = init_adam_state(params)
    
    # Initialize Env State (Vectorized)
    key, *env_keys = jax.random.split(key, num_envs + 1)
    env_states = vmap(env.reset)(jnp.array(env_keys))
    episode_returns = jnp.zeros(num_envs)
    returned_episodic_returns = jnp.zeros(num_envs)
    
    # --- JIT Compiled Update Step (One Segment) ---
    
    @jit
    def train_segment(carry, _):
        params, opt_state, env_states, episode_returns, key = carry
        
        # --- 1. Rollout Collection ---
        def rollout_step(carry, _):
            env_states, episode_returns, key = carry
            key, subkey = jax.random.split(key)
            
            # Action selection
            logits = get_action_logits(params['actor'], env_states)
            action = jax.random.categorical(subkey, logits)
            log_prob = jax.nn.log_softmax(logits)
            action_log_prob = jnp.take_along_axis(log_prob, action[:, None], axis=1).squeeze(-1)
            value = get_value(params['critic'], env_states)
            
            # Env step
            next_env_states, reward, done = vmap(env.step)(env_states, action)
            
            # Auto-reset
            key, *reset_keys = jax.random.split(key, num_envs + 1)
            reset_states = vmap(env.reset)(jnp.array(reset_keys))
            next_env_states = jnp.where(done[:, None], reset_states, next_env_states)
            
            # Update episodic returns
            episode_returns = episode_returns + reward
            final_return = jnp.where(done, episode_returns, 0.0)
            episode_returns = jnp.where(done, 0.0, episode_returns)
            
            transition = (env_states, action, action_log_prob, reward, done, value, final_return)
            return (next_env_states, episode_returns, key), transition

        (next_env_states, episode_returns, key), traj = jax.lax.scan(
            rollout_step, (env_states, episode_returns, key), None, length=num_steps
        )
        
        obs, actions, logprobs, rewards, dones, values, final_returns = traj
        
        # --- 2. GAE Calculation ---
        next_value = get_value(params['critic'], next_env_states)
        
        def gae_scan(carry, t):
            last_gae_lam, next_val = carry
            delta = rewards[t] + args.gamma * next_val * (1.0 - dones[t]) - values[t]
            last_gae_lam = delta + args.gamma * args.gae_lambda * (1.0 - dones[t]) * last_gae_lam
            return (last_gae_lam, values[t]), last_gae_lam

        _, advantages = jax.lax.scan(
            gae_scan, 
            (jnp.zeros_like(next_value), next_value), 
            jnp.arange(num_steps), 
            reverse=True
        )
        returns = advantages + values
        
        # Flatten batch
        b_obs = obs.reshape((batch_size, -1))
        b_logprobs = logprobs.reshape(batch_size)
        b_actions = actions.reshape(batch_size)
        b_advantages = advantages.reshape(batch_size)
        
        # Normalize advantages (Batch level)
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
        b_returns = returns.reshape(batch_size)
        b_values = values.reshape(batch_size)
        
        # --- 3. PPO Update ---
        def update_epoch(carry, _):
            params, opt_state, key = carry
            key, subkey = jax.random.split(key)
            inds = jax.random.permutation(subkey, batch_size)
            
            def update_minibatch(carry, start_idx):
                params, opt_state = carry
                
                # Dynamic slice for minibatches
                def get_minibatch(x):
                    if x.ndim == 1:
                        return jax.lax.dynamic_slice(x, (start_idx,), (minibatch_size,))
                    else:
                        return jax.lax.dynamic_slice(x, (start_idx, 0), (minibatch_size, x.shape[1]))

                mb_obs = get_minibatch(b_obs)
                mb_logprobs = get_minibatch(b_logprobs)
                mb_actions = get_minibatch(b_actions)
                mb_advantages = get_minibatch(b_advantages)
                mb_returns = get_minibatch(b_returns)
                mb_values = get_minibatch(b_values) # <--- Added missing extraction
                
                def loss_fn(params):
                    new_logits = get_action_logits(params['actor'], mb_obs)
                    new_values = get_value(params['critic'], mb_obs)
                    
                    new_logprobs_all = jax.nn.log_softmax(new_logits)
                    new_logprobs = jnp.take_along_axis(new_logprobs_all, mb_actions[:, None], axis=1).squeeze(-1)
                    entropy = -jnp.sum(jax.nn.softmax(new_logits) * new_logprobs_all, axis=1).mean()
                    
                    logratio = new_logprobs - mb_logprobs
                    ratio = jnp.exp(logratio)
                    
                    # Advantage normalization is now done at batch level
                    advs = mb_advantages
                        
                    pg_loss = -jnp.minimum(
                        advs * ratio, 
                        advs * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    ).mean()
                    
                    if args.clip_vloss:
                        v_loss_unclipped = (new_values - mb_returns) ** 2
                        v_clipped = mb_values + jnp.clip(
                            new_values - mb_values,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                    loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss
                    return loss, (pg_loss, v_loss, entropy)

                (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(params)
                params, opt_state = adam_update(grads, opt_state, params, args.learning_rate, args.max_grad_norm)
                return (params, opt_state), metrics

            (params, opt_state), batch_metrics = jax.lax.scan(
                update_minibatch, (params, opt_state), 
                jnp.arange(0, batch_size, minibatch_size)
            )
            return (params, opt_state, key), batch_metrics

        (params, opt_state, key), epoch_metrics = jax.lax.scan(
            update_epoch, (params, opt_state, key), None, length=args.update_epochs
        )
        
        # Metrics
        metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)
        
        # Calculate average episodic return (only for completed episodes)
        # We sum all final returns and divide by the number of done episodes
        # If no episodes finished, we return NaN (or 0, but NaN is safer to detect "no data")
        num_dones = dones.sum()
        avg_return = jnp.where(
            num_dones > 0,
            final_returns.sum() / num_dones,
            0.0 # Default to 0 if no episodes finished this segment
        )
        
        return (params, opt_state, next_env_states, episode_returns, key), (metrics, avg_return)

    # --- Main Training Loop ---
    
    print(f"Starting training for {total_iterations} iterations...")
    start_time = time.perf_counter()
    
    # Loop over segments (checkpoints)
    current_iter = 0
    # pbar = tqdm(total=total_iterations)
    
    while current_iter < total_iterations:
        segment_start_time = time.perf_counter()
        # Determine how many iterations to run in this segment
        next_checkpoint = min(current_iter + checkpoint_freq, total_iterations)
        segment_length = next_checkpoint - current_iter
        
        # Run JIT-compiled segment
        (params, opt_state, env_states, episode_returns, key), (metrics, avg_returns) = jax.lax.scan(
            train_segment, 
            (params, opt_state, env_states, episode_returns, key), 
            None, 
            length=segment_length
        )
        
        # Wait for computation
        jax.block_until_ready(params)
        
        # Update progress
        current_iter += segment_length
        # pbar.update(segment_length)
        
        # Get latest metrics (from the last step of the scan)
        # We want the last non-zero return if possible, but for now just taking the last one is fine
        # Note: If no episodes finished in the last segment, this might be 0.
        last_return = avg_returns[-1]
        # pbar.set_postfix({"return": f"{last_return:.2f}"})

        # Calculate SPS
        current_time = time.perf_counter()
        segment_time = current_time - segment_start_time
        sps = (segment_length * batch_size) / segment_time
        
        # Calculate Elapsed & ETA
        elapsed = current_time - start_time
        # Avoid division by zero
        if current_iter > 0:
            eta = (total_iterations - current_iter) * (elapsed / current_iter)
        else:
            eta = 0
            
        # Metrics
        # metrics is (pg_loss, v_loss, entropy)
        # We need to take the mean over the segment (scan dimension)
        pg_loss = metrics[0].mean().item()
        v_loss = metrics[1].mean().item()
        entropy = metrics[2].mean().item()
        
        print(f"Update: {current_iter}/{total_iterations}, Steps: {current_iter * batch_size}, "
              f"Return: {last_return:.2f}, SPS: {int(sps)}, "
              f"Elapsed: {format_time(elapsed)}, ETA: {format_time(eta)}, "
              f"PLoss: {pg_loss:.3f}, VLoss: {v_loss:.3f}, Ent: {entropy:.3f}")

        
        # Video Recording
        if args.capture_video:
            record_video(params, env, run_name, current_iter * batch_size)

    end_time = time.perf_counter()
    total_steps = total_iterations * batch_size
    sps = total_steps / (end_time - start_time)
    print(f"\nTraining finished in {end_time - start_time:.2f}s")
    print(f"SPS: {sps:.2f}")

def record_video(params, env, run_name, step):
    frames = []
    # Use a fresh key for video
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    
    for _ in range(500):
        # Render
        frame = render_cartpole(np.array(state))
        frames.append(frame)
        
        # Step
        logits = get_action_logits(params['actor'], state)
        action = jnp.argmax(logits)
        state, _, done = env.step(state, action)
        
        if done:
            break
            
    video_dir = f"videos/{run_name}"
    os.makedirs(video_dir, exist_ok=True)
    video_path = f"{video_dir}/step_{step}.mp4"
    imageio.mimsave(video_path, frames, fps=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--num-envs", type=int, default=64) # Reduced for better sample efficiency
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--clip-vloss", action="store_true", default=True)
    parser.add_argument("--capture-video", action="store_true", default=True)
    parser.add_argument("--checkpoint", type=int, default=10, help="Frequency of video recording (in iterations)")
    
    args = parser.parse_args()
    train(args)