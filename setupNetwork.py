import jax
import flax.linen as nn
import swarmrl as srl
import optax
from jax import numpy as jnp
import logging
import time
import os

logger = logging.getLogger(__name__)
action_dimension = 6
action_limits = jnp.array([[0,70],[0,70],[0,30], [0,30], [-0.8, 0.8], [-0.5, 0.5]])




import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple


class ParticlePreprocessor(nn.Module):
    """DeepSets + GCN‐style preprocessor:
       1) Per‐particle MLP (shared),
       2) Mean pooling (DeepSets),
       3) One GCN layer over fully connected graph,
       4) Final concatenation with velocity features.
    """
    hidden_dim: int = 64
    gcn_hidden: int = 64
    num_particles: int = 30         # total particles (including velocity slots)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self,
                 state: jnp.ndarray,
                 train: bool = False) -> jnp.ndarray:
        """
        Args:
          state: shape (batch, time, n_particles, dim)
                 (e.g. n_particles=32 if last 2 are velocity “particles”)
        Returns:
          features: shape (batch, time, hidden_dim + velocity_dim)
        """
        # ────────────────────────────────────────────────────────────────────────
        # 1) Split positions vs. velocity “particles”
        #    (Here we assume the last 2 “particles” encode velocity in some form.)
        pos = state[:, :, :-2, :]    # shape (B, T, Npos= n_p-2, Dpos)
        vel = state[:, :, -2:, :]    # shape (B, T, 2, Dvel)
        b, t, n_pos, d_pos = pos.shape

        # ────────────────────────────────────────────────────────────────────────
        # 2) DeepSets per‐particle MLP (shared):
        #    Apply same small MLP to each particle
        def per_particle_mlp(x):
            x = nn.Dense(self.hidden_dim,
                         kernel_init=nn.initializers.kaiming_normal())(x)
            x = nn.silu(x)
            x = nn.Dense(self.hidden_dim,
                         kernel_init=nn.initializers.kaiming_normal())(x)
            x = nn.silu(x)
            x = nn.LayerNorm()(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
            return x

        # Flatten batch & time so we can vmap over particles easily
        x = pos.reshape(b * t, n_pos, d_pos)   # (B*T, Npos, Dpos)
        # Apply shared MLP to each particle
        x = jax.vmap(per_particle_mlp, in_axes=1, out_axes=1)(x)
        # Now x has shape (B*T, Npos, hidden_dim)

        # ────────────────────────────────────────────────────────────────────────
        # 3) DeepSets pooling (mean over particles)
        x_mean = jnp.mean(x, axis=1)           # (B*T, hidden_dim)

        # ────────────────────────────────────────────────────────────────────────
        # 4) One GCN layer over fully connected graph of particles
        #    We build a simple adjacency: all pairwise edges (no self‐loops)
        #    GCN update: h_i' = σ( Σ_j W_edge * h_j + W_self * h_i )
        h = x  # (B*T, Npos, hidden_dim)
        W_edge = self.param("W_edge",
                            nn.initializers.kaiming_normal(),
                            (self.hidden_dim, self.gcn_hidden))
        W_self = self.param("W_self",
                            nn.initializers.kaiming_normal(),
                            (self.hidden_dim, self.gcn_hidden))
        # Expand so we can do pairwise sums
        h_i = jnp.expand_dims(h, 2)            # (B*T, Npos, 1, hidden_dim)
        h_j = jnp.expand_dims(h, 1)            # (B*T, 1, Npos, hidden_dim)
        # Sum neighbor features (including all j ≠ i)
        neighbor_sum = jnp.sum(h_j, axis=2)    # (B*T, Npos, hidden_dim)
        # GCN linear transforms
        out_edge = neighbor_sum @ W_edge        # (B*T, Npos, gcn_hidden)
        out_self = h @ W_self                   # (B*T, Npos, gcn_hidden)
        h_gcn = nn.silu(out_edge + out_self)
        h_gcn = nn.LayerNorm()(h_gcn)

        # Pool GCN outputs similarly (mean)
        h_gcn_mean = jnp.mean(h_gcn, axis=1)   # (B*T, gcn_hidden)

        # ────────────────────────────────────────────────────────────────────────
        # 5) Combine DeepSets + GCN pooled features
        combined = jnp.concatenate([x_mean, h_gcn_mean], axis=-1)  # (B*T, hidden_dim + gcn_hidden)
        combined = nn.Dense(self.hidden_dim,
                            kernel_init=nn.initializers.kaiming_normal())(combined)
        combined = nn.silu(combined)
        combined = nn.LayerNorm()(combined)
        combined = nn.Dropout(rate=self.dropout_rate)(combined, deterministic=not train)
        combined = combined.reshape(b, t, -1)   # (B, T, hidden_dim)

        # ────────────────────────────────────────────────────────────────────────
        # 6) Process velocity features (shared MLP)
        v = vel.reshape(b * t, -1)              # (B*T, 2 * Dvel)
        v = nn.Dense(16,
                     kernel_init=nn.initializers.kaiming_normal())(v)
        v = nn.silu(v)
        v = nn.LayerNorm()(v)
        v = nn.Dropout(rate=self.dropout_rate)(v, deterministic=not train)
        v = v.reshape(b, t, -1)                 # (B, T, 16)

        # ────────────────────────────────────────────────────────────────────────
        # 7) Final concatenation
        return jnp.concatenate([combined, v], axis=-1)  # (B, T, hidden_dim + 16)


class ActorNet(nn.Module):
    """SAC Gaussian actor with DeepSets+GCN preprocessor."""
    preprocessor: Any          # ParticlePreprocessor(shared)
    hidden_dims: Tuple[int, ...] = (128, 64,32)
    log_std_min: float = -10.0
    log_std_max: float = 0.5
    dropout_rate: float = 0.1

    
    def setup(self):
        self.ScanLSTM = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        self.lstm = self.ScanLSTM(features=2)
        self.temperature = self.param("temperature", lambda key, shape: jnp.full(shape, jnp.log(1)), (1,))

    @nn.compact
    def __call__(self,
                 state: jnp.ndarray,
                 previous_actions: jnp.ndarray,
                 carry: Any = None,
                 train: bool = False) -> Tuple[jnp.ndarray, Any]:
        if carry is None:
            carry = self.lstm.initialize_carry(
                jax.random.PRNGKey(0), state.shape[:1] + state.shape[2:]
            )
        # 1) Preprocess (b, t, hidden_dim + 16)
        x = self.preprocessor(state / 253.0, train=train)
        b, t, f = x.shape
        x = x.reshape(b, t * f)   # flatten time & features

        # 2) MLP with Dropout & L2 on weights
        for hd in self.hidden_dims:
            x = nn.Dense(
                hd,
                kernel_init=nn.initializers.kaiming_normal(),
                bias_init=nn.initializers.zeros
            )(x)
            x = nn.silu(x)
            x = nn.LayerNorm()(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        
        # 3) Output mu & log_std
        mu = nn.Dense(
            action_dimension,
            kernel_init=nn.initializers.orthogonal(1e-2),
            bias_init=nn.initializers.zeros,
        )(x)
        mu = jnp.tanh(mu) * 3.0    # squash into [-3, 3] for example

        log_std = nn.Dense(
            action_dimension,
            kernel_init=nn.initializers.orthogonal(1e-2),
            bias_init=nn.initializers.zeros,
        )(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)

        out = jnp.concatenate([mu, log_std], axis=-1)  # (b, action_dim*2)
        y = nn.BatchNorm(use_running_average=not train)(out)
        return out, carry
    
class CriticNet(nn.Module):
    """Twin‐Q SAC critic with DeepSets+GCN preprocessor."""
    preprocessor: Any
    hidden_dims: Tuple[int, ...] = (128, 64,32)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self,
                 state: jnp.ndarray,
                 previous_actions: jnp.ndarray,
                 action: jnp.ndarray,
                 carry: Any = None,
                 train: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # 1) Preprocess
        x = self.preprocessor(state / 253.0, train=train)  # (b, t, f)
        b, t, f = x.shape
        x = x.reshape(b, t * f)                             # (b, t*f)

        # 2) Normalize action to [−1,1] or similar
        a_norm = (action - action_limits[:, 0]) / (action_limits[:, 1] - action_limits[:, 0])
        sa = jnp.concatenate([x, a_norm], axis=-1)          # (b, t*f + action_dim)

        # 3) Twin Q‐networks with shared MLP trunk, then separate heads
        def q_branch(name: str):
            y = sa
            for hd in self.hidden_dims:
                y = nn.Dense(
                    hd,
                    kernel_init=nn.initializers.kaiming_normal(),
                    bias_init=nn.initializers.zeros,
                    name=f"{name}_fc{hd}"
                )(y)
                y = nn.silu(y)
                y = nn.LayerNorm()(y)
                y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=not train)
            q = nn.Dense(
                1,
                kernel_init=nn.initializers.orthogonal(1e-2),
                bias_init=nn.initializers.zeros,
                name=f"{name}_out"
            )(y)
            return q

        q1 = q_branch("q1")
        q2 = q_branch("q2")
        z = nn.BatchNorm(use_running_average=not train)(q1)
        return q1, q2

    
def defineRLAgent(
    obs, task: srl.tasks.Task, learning_rate: float, resolution: int=506, sequence_length: int=4, number_particles: int = 7, lock=None
) -> srl.agents.MPIActorCriticAgent:
    # Define the model
    
    if learning_rate == 0.0:
        logger.info("Deployment mode")
        optimizer = optax.adam(learning_rate=0.0)
    else:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=10000,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping with a maximum norm of 1.0
            optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule)
        )


    shared_encoder = ParticlePreprocessor()
    actor  = ActorNet(preprocessor=shared_encoder)
    critic = CriticNet(preprocessor=shared_encoder)

    

    # Define a sampling_strategy
    sampling_strategy = srl.sampling_strategies.ContinuousGaussianDistribution(action_dimension=action_dimension, action_limits=action_limits)
    exploration_policy = srl.exploration_policies.GlobalOUExploration(
        drift=0.1, volatility=0.08, action_dimension=action_dimension, action_limits=action_limits
    )

    value_function = srl.value_functions.TDReturnsSAC(gamma=0.99, standardize=False)
    actor_network = srl.networks.ContinuousActionModel(
        flax_model=actor,
        optimizer=optimizer,
        input_shape=(
            1,
            sequence_length,
            number_particles + 4,
            2,
        ),  # batch implicitly 1 ,time,H,W,channels for conv
        sampling_strategy=sampling_strategy,
        exploration_policy=exploration_policy,
        action_dimension=action_dimension,
        deployment_mode=learning_rate == 0.0,
        rng_key=jax.random.PRNGKey(int(time.time())),
    )
    critic_network = srl.networks.ContinuousCriticModel(
        critic_model=critic,
        optimizer=optimizer,
        input_shape=(
            1,
            sequence_length,
            number_particles + 4,
            2,
        ),  # batch implicitly 1 ,time,H,W,channels for conv
        action_dimension=action_dimension,
        rng_key=jax.random.PRNGKey(int(time.time())),
    )

    loss = srl.losses.SoftActorCriticGradientLoss(
        value_function=value_function,
        minimum_entropy=-action_dimension*1.2,
        polyak_averaging_tau=0.02,
        lock=lock,
        validation_split=0.1,
        fix_temperature=False,
        batch_size=1012,
    )

    protocol = srl.agents.MPIActorCriticAgent(
        particle_type=0,
        actor_network=actor_network,
        critic_network=critic_network,
        task=task,
        observable=obs,
        loss=loss,
        max_samples_in_trajectory=1000,
        lock=lock
    )
    return protocol
