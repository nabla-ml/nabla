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


# --- Setup ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --- 1. The Environment ---

class PureJaxCartPoleSwingUp:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masscart + self.masspole)
        self.length = 0.5 
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 20.0 
        self.tau = 0.02 
        self.x_threshold = 2.4 
        
    # def reset(self, key):
    #     k1, k2, k3, k4 = jax.random.split(key, 4)
    #     x = jax.random.uniform(k1, minval=-0.2, maxval=0.2)
    #     x_dot = jax.random.uniform(k2, minval=-0.2, maxval=0.2)
    #     # Start hanging DOWN (pi)
    #     theta = jnp.pi + jax.random.uniform(k3, minval=-0.2, maxval=0.2)
    #     theta_dot = jax.random.uniform(k4, minval=-0.2, maxval=0.2)
    #     return jnp.array([x, x_dot, theta, theta_dot])

    def reset(self, key):
        k_x, k_theta, k_vel = jax.random.split(key, 3)
        
        x = jax.random.uniform(k_x, minval=-1.0, maxval=1.0)
        
        # Start the pole at ANY angle (0 to 2pi)
        # It might spawn upright, upside down, or sideways.
        theta = jax.random.uniform(k_theta, minval=0, maxval=2*jnp.pi)
        
        return jnp.array([x, 0.0, theta, 0.0])

    def get_obs(self, state):
        x, x_dot, theta, theta_dot = state
        # Normalization hint: Scale raw values roughly to [-1, 1] range for the neural net
        return jnp.array([
            x / 2.4, 
            x_dot / 2.0, 
            jnp.cos(theta), 
            jnp.sin(theta), 
            theta_dot / 3.0
        ])

    def step(self, state, action):
        x, x_dot, theta, theta_dot = state
        
        # Action is raw network output. We clip it here for physics safety.
        # Note: We do NOT tanh the action in the network, we clip it here.
        # This prevents gradient saturation.
        force = jnp.clip(action[0], -1.0, 1.0) * self.force_mag
        
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        next_state = jnp.array([x, x_dot, theta, theta_dot])

        # --- Conditional Position Penalty (Balanced) ---
        # KEY: Position penalty depends on pole state
        # - During swing-up: light penalty (allow exploration)
        # - When balanced: moderate penalty (allow learning, prevent drift)
        
        pole_height = jnp.cos(theta)
        r_height = pole_height
        
        # Upright bonus
        upright_bonus = 1.5 * jnp.exp(-10.0 * (1.0 - pole_height)**2)
        
        # Balanced position penalty
        is_upright = (pole_height > 0.95).astype(jnp.float32)
        r_pos = -0.01 * (x**2) - 0.3 * (x**2) * is_upright
        # At x=2.0 when upright: -0.01*4 - 0.3*4 = -1.24 (still < +1.5 bonus)
        # But at x=0.5: -0.085 (allows learning without harsh punishment)
        
        # Velocity penalty when upright
        r_vel = -0.05 * (theta_dot**2) * is_upright
        
        # Action penalty
        r_action = -0.0005 * (action[0]**2)
        
        reward = r_height + upright_bonus + r_pos + r_vel + r_action

        done = (x < -self.x_threshold) | (x > self.x_threshold)
        
        return next_state, reward, done

# --- 2. Renderer ---
def render_cartpole(state, width=600, height=400):
    x, _, theta, _ = state
    world_width = 2.4 * 2
    scale = width / world_width
    carty = 250 
    polelen = scale * 1.0 
    
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(0, carty), (width, carty)], fill=(0, 0, 0), width=1)
    
    cartx = x * scale + width / 2.0
    draw.rectangle([cartx-25, carty-15, cartx+25, carty+15], fill=(0, 0, 0))

    # In PIL, Y is down. We want theta=0 to be UP (Negative Y).
    # sin(0)=0, -cos(0)=-1. Correct.
    tip_x = cartx + polelen * np.sin(theta)
    tip_y = carty - polelen * np.cos(theta)
    
    draw.line([(cartx, carty), (tip_x, tip_y)], fill=(204, 153, 102), width=12)
    draw.ellipse([cartx - 5, carty - 5, cartx + 5, carty + 5], fill=(127, 127, 255))
    return np.array(img)

# --- 3. Neural Networks (Wider & Better Init) ---

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

def init_actor_critic(key, obs_dim, action_dim, hidden_dim=512): # INCREASED SIZE
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    return {
        'actor': {
            'l1': init_mlp_layer(k1, obs_dim, hidden_dim, scale=np.sqrt(2)),
            'l2': init_mlp_layer(k2, hidden_dim, hidden_dim, scale=np.sqrt(2)),
            'mean': init_mlp_layer(k3, hidden_dim, action_dim, scale=0.01),
            # Init log_std to -0.5 (approx 0.6 std). 
            # High enough to explore, low enough not to be random noise.
            'log_std': jnp.full((action_dim,), -0.5) 
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
    return x

def get_action_dist(actor_params, obs):
    x = forward_mlp(actor_params, obs)
    mean = x @ actor_params['mean']['w'] + actor_params['mean']['b']
    # Removed tanh here. We want unbounded mean for the Gaussian.
    # The environment will clip it.
    # CRITICAL: Clamp log_std to prevent policy collapse
    # std will be in range [0.135, 1.65]
    log_std = jnp.clip(actor_params['log_std'], -2.0, 0.5)
    return mean, log_std

def get_value(critic_params, obs):
    x = forward_mlp(critic_params, obs)
    return (x @ critic_params['head']['w'] + critic_params['head']['b']).squeeze(-1)

# --- 4. Optimizer ---
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

# --- 5. Training Logic ---

def train(args):
    run_name = f"JaxSwingUp_Stable_{int(time.time())}"
    print(f"Running STABLE JAX CartPole Swing-Up: {run_name}")
    
    num_envs = args.num_envs
    num_steps = args.num_steps
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // args.num_minibatches
    total_iterations = args.total_timesteps // batch_size
    
    env = PureJaxCartPoleSwingUp()
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    
    # Larger Hidden Dim
    params = init_actor_critic(init_key, obs_dim=5, action_dim=1, hidden_dim=512)
    opt_state = init_adam_state(params)
    
    key, *env_keys = jax.random.split(key, num_envs + 1)
    env_states = vmap(env.reset)(jnp.array(env_keys))
    episode_returns = jnp.zeros(num_envs)
    
    @jit
    def train_segment(carry, update_i): # Added update_i for LR annealing
        params, opt_state, env_states, episode_returns, key = carry
        
        # --- 1. Rollout ---
        def rollout_step(carry, step_idx):
            env_states, episode_returns, episode_lengths, key = carry
            key, subkey = jax.random.split(key)
            obs = vmap(env.get_obs)(env_states)
            
            mean, log_std = get_action_dist(params['actor'], obs)
            std = jnp.exp(log_std)
            noise = jax.random.normal(subkey, mean.shape)
            action = mean + std * noise
            
            # Simple Gaussian Log Prob
            action_log_prob = -0.5 * ((action - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
            action_log_prob = action_log_prob.sum(axis=-1)
            
            value = get_value(params['critic'], obs)
            next_env_states, reward, done_boundary = vmap(env.step)(env_states, action)
            
            # Episode length limit: 500 steps
            episode_lengths = episode_lengths + 1
            done_length = episode_lengths >= 500
            done = done_boundary | done_length
            
            key, *reset_keys = jax.random.split(key, num_envs + 1)
            reset_states = vmap(env.reset)(jnp.array(reset_keys))
            next_env_states = jnp.where(done[:, None], reset_states, next_env_states)
            
            episode_returns = episode_returns + reward
            final_return = jnp.where(done, episode_returns, 0.0)
            episode_returns = jnp.where(done, 0.0, episode_returns)
            episode_lengths = jnp.where(done, 0, episode_lengths)
            
            transition = (obs, action, action_log_prob, reward, done, value, final_return)
            return (next_env_states, episode_returns, episode_lengths, key), transition

        (next_env_states, episode_returns, episode_lengths, key), traj = jax.lax.scan(
            rollout_step, (env_states, episode_returns, jnp.zeros(num_envs, dtype=jnp.int32), key), None, length=num_steps
        )
        obs, actions, logprobs, rewards, dones, values, final_returns = traj
        
        # --- 2. GAE ---
        next_obs = vmap(env.get_obs)(next_env_states)
        next_value = get_value(params['critic'], next_obs)
        
        def gae_scan(carry, t):
            last_gae_lam, next_val = carry
            delta = rewards[t] + args.gamma * next_val * (1.0 - dones[t]) - values[t]
            last_gae_lam = delta + args.gamma * args.gae_lambda * (1.0 - dones[t]) * last_gae_lam
            return (last_gae_lam, values[t]), last_gae_lam

        _, advantages = jax.lax.scan(gae_scan, (jnp.zeros_like(next_value), next_value), jnp.arange(num_steps), reverse=True)
        returns = advantages + values
        
        # Flatten
        b_obs = obs.reshape((batch_size, -1))
        b_logprobs = logprobs.reshape(batch_size)
        b_actions = actions.reshape((batch_size, -1))
        b_advantages = advantages.reshape(batch_size)
        b_returns = returns.reshape(batch_size)
        b_values = values.reshape(batch_size)
        
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
            
        # --- 3. Update ---
        # LINEAR LR ANNEALING
        frac = 1.0 - (update_i / total_iterations)
        current_lr = args.learning_rate * frac

        def update_epoch(carry, _):
            params, opt_state, key = carry
            key, subkey = jax.random.split(key)
            inds = jax.random.permutation(subkey, batch_size)
            
            sb_obs, sb_logprobs, sb_actions = b_obs[inds], b_logprobs[inds], b_actions[inds]
            sb_advantages, sb_returns, sb_values = b_advantages[inds], b_returns[inds], b_values[inds]

            def process_minibatch(carry, i):
                params, opt_state = carry
                start = i * minibatch_size
                # Slicing...
                mb_obs = jax.lax.dynamic_slice(sb_obs, (start, 0), (minibatch_size, sb_obs.shape[1]))
                mb_logprobs = jax.lax.dynamic_slice(sb_logprobs, (start,), (minibatch_size,))
                mb_actions = jax.lax.dynamic_slice(sb_actions, (start, 0), (minibatch_size, sb_actions.shape[1]))
                mb_advantages = jax.lax.dynamic_slice(sb_advantages, (start,), (minibatch_size,))
                mb_returns = jax.lax.dynamic_slice(sb_returns, (start,), (minibatch_size,))
                mb_values = jax.lax.dynamic_slice(sb_values, (start,), (minibatch_size,))

                def loss_fn(params):
                    mean, log_std = get_action_dist(params['actor'], mb_obs)
                    std = jnp.exp(log_std)
                    new_logprobs = -0.5 * ((mb_actions - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
                    new_logprobs = new_logprobs.sum(axis=-1)
                    
                    entropy = (log_std + 0.5 + 0.5 * jnp.log(2 * jnp.pi)).sum(axis=-1).mean()
                    
                    logratio = new_logprobs - mb_logprobs
                    ratio = jnp.exp(logratio)
                    
                    pg_loss = -jnp.minimum(
                        mb_advantages * ratio, 
                        mb_advantages * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    ).mean()
                    
                    new_values = get_value(params['critic'], mb_obs)
                    v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                    
                    loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss
                    return loss, (pg_loss, v_loss, entropy)

                (loss, metrics), grads = value_and_grad(loss_fn, has_aux=True)(params)
                # Pass current_lr here
                params, opt_state = adam_update(grads, opt_state, params, current_lr, args.max_grad_norm)
                return (params, opt_state), metrics

            (params, opt_state), batch_metrics = jax.lax.scan(process_minibatch, (params, opt_state), jnp.arange(args.num_minibatches))
            return (params, opt_state, key), batch_metrics

        (params, opt_state, key), epoch_metrics = jax.lax.scan(update_epoch, (params, opt_state, key), None, length=args.update_epochs)
        
        # Return metrics
        num_dones = dones.sum()
        # Return average of returns for completed episodes (if any), otherwise 0
        avg_ret = jnp.where(num_dones > 0, final_returns.sum() / num_dones, 0.0)
        
        # Calculate mean metrics for the epoch
        metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)
        
        return (params, opt_state, next_env_states, episode_returns, key), (avg_ret, metrics)


    # --- Training Loop ---
    print(f"Starting training for {total_iterations} iterations...")
    # pbar = tqdm(total=total_iterations)
    moving_avg_ret = 0.0
    start_time = time.perf_counter()
    last_time = start_time

    
    for i in range(total_iterations):
        # Pass i (update index) to train_segment for annealing
        (params, opt_state, env_states, episode_returns, key), (avg_ret, metrics) = train_segment(
            (params, opt_state, env_states, episode_returns, key), i
        )
        jax.block_until_ready(params)
        
        if avg_ret != 0:
            moving_avg_ret = 0.95 * moving_avg_ret + 0.05 * avg_ret if moving_avg_ret != 0 else avg_ret

        # pbar.update(1)
        # pbar.set_postfix({"Ret": f"{moving_avg_ret:.2f}"})
        
        if i % args.checkpoint == 0 or i == total_iterations - 1:
            current_time = time.perf_counter()
            sps = (args.checkpoint * batch_size) / (current_time - last_time)
            last_time = current_time
            
            elapsed = current_time - start_time
            if i > 0:
                eta = (total_iterations - i) * (elapsed / i)
            else:
                eta = 0
            
            # Unpack metrics
            pg_loss, v_loss, entropy = metrics
            
            print(f"Update: {i}/{total_iterations}, Steps: {i * batch_size}, "
                  f"Return: {moving_avg_ret:.2f}, SPS: {int(sps)}, "
                  f"Elapsed: {format_time(elapsed)}, ETA: {format_time(eta)}, "
                  f"PLoss: {pg_loss.item():.3f}, VLoss: {v_loss.item():.3f}, Ent: {entropy.item():.3f}")

        
        if args.capture_video and i % args.checkpoint == 0 and i > 0:
            record_video(params, env, run_name, i * batch_size)

# --- 6. Video Recording (THE FIX) ---
def record_video(params, env, run_name, step):
    frames = []
    # Use deterministic key for easier comparison
    key = jax.random.PRNGKey(42) 
    state = env.reset(key)
    
    for _ in range(800):
        frames.append(render_cartpole(np.array(state)))
        
        obs = env.get_obs(state)
        # Use mean for deterministic evaluation video
        mean, _ = get_action_dist(params['actor'], obs)
        action = mean 
        state, _, done = env.step(state, action)
        
        if done: break
            
    video_dir = f"videos/{run_name}"
    os.makedirs(video_dir, exist_ok=True)
    imageio.mimsave(f"{video_dir}/step_{step}.mp4", frames, fps=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)  # Slightly lower for stability
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.995)  # Higher for long-horizon swing-up
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)  # 5x higher for exploration
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)  # Higher for better gradient flow
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--capture-video", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=int, default=5) # Save less frequently
    
    args = parser.parse_args()
    train(args)