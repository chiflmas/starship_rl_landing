import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from policy_warmup_sac import PolicyWarmupSAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    VecVideoRecorder,
    sync_envs_normalization,
)
from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps, BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch
import multiprocessing as mp
import random
from rocket import Rocket
from curriculum_phases import CURRICULUM_PHASES, DEFAULT_CURRICULUM_PHASE
import os
import sys
from importlib import import_module
from collections import defaultdict
from checkpointing import CheckpointStore, LandingEvalCallback, PeriodicCheckpointCallback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

VERSION_CONFIGS = {
    "v1": {
        "description": "Single actuator with 12 observations",
        "engine_mode": "single",
        "observation_mode": "v1_engineered",
        "observation_dims": 12,
        "reward_module": "improved_rewards_v1",
        "reward_class": "V1SingleEngineRewardSystem",
        "output_name": "v1_single_engine",
    },
    "v1_1": {
        "description": "Single actuator with 10 raw observations",
        "engine_mode": "single",
        "observation_mode": "v1_raw",
        "observation_dims": 10,
        "reward_module": "improved_rewards_v1_1",
        "reward_class": "V11SingleEngineRewardSystem",
        "output_name": "v1_1_single_engine",
    },
    "v2": {
        "description": "Three engines, 17 compact observations, and 9 actions",
        "engine_mode": "three_v2",
        "observation_mode": "v2_raw",
        "observation_dims": 17,
        "reward_module": "improved_rewards_v2",
        "reward_class": "V2ThreeEngineRewardSystem",
        "output_name": "v2_three_engine_compact17",
    },
    "legacy_three_engine": {
        "description": "Recovered three-engine environment",
        "engine_mode": "three",
        "observation_mode": "legacy",
        "observation_dims": 23,
        "reward_module": "improved_rewards",
        "reward_class": "ImprovedRewardSystem",
        "output_name": "legacy_three_engine",
    },
}

VERSION_ALIASES = {
    "v1_single_engine": "v1",
    "v1_1_single_engine": "v1_1",
}

DEFAULT_VERSION = "v1"


def resolve_version(version):
    """Resolve a public version name and return an isolated config copy."""
    canonical_version = VERSION_ALIASES.get(version, version)
    if canonical_version not in VERSION_CONFIGS:
        available = ", ".join(VERSION_CONFIGS)
        raise ValueError(
            f"Unknown environment version {version!r}. Available: {available}"
        )
    return canonical_version, dict(VERSION_CONFIGS[canonical_version])


def load_reward_class(version_config):
    module = import_module(version_config["reward_module"])
    return getattr(module, version_config["reward_class"])


def set_reproducible_seed(seed):
    """Configura las fuentes de aleatoriedad usadas por el entrenamiento."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ═══════════════════════════════════════════════════════════════
# STARSHIP GYM ENVIRONMENT (Mejorado para SAC)
# ═══════════════════════════════════════════════════════════════

class StarshipGymEnv(gym.Env):
    def __init__(self, task='landing', max_steps=750, render_mode=None,
                 version=DEFAULT_VERSION,
                 curriculum_phase=DEFAULT_CURRICULUM_PHASE,
                 render_layout='default'):
        super().__init__()
        
        self.version, self.version_config = resolve_version(version)
        self.engine_mode = self.version_config['engine_mode']
        self.render_mode = render_mode
        self.render_layout = render_layout
        self.curriculum_phase = curriculum_phase
        self.rocket_env = Rocket(
            task=task,
            max_steps=max_steps,
            rocket_type='starship',
            engine_mode=self.engine_mode,
            curriculum_phase=curriculum_phase,
            observation_mode=self.version_config['observation_mode'],
        )

        reward_class = load_reward_class(self.version_config)

        self.rocket_env.reward_system = reward_class(
            task=task,
            world_bounds={
                'x_min': self.rocket_env.world_x_min,
                'x_max': self.rocket_env.world_x_max,
                'y_min': self.rocket_env.world_y_min,
                'y_max': self.rocket_env.world_y_max
            }
        )

        # Rocket.flatten() devuelve state_dims componentes. Mantener este
        # espacio ligado al entorno evita incompatibilidades con VecNormalize.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.rocket_env.state_dims,),
            dtype=np.float32,
        )
        if self.engine_mode == 'single':
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.engine_mode == 'three_v2':
            # V2 separates throttle, gimbal, and ON/OFF for each engine. This
            # removes the discontinuity that previously placed minimum thrust
            # next to the shutdown boundary in the same action component.
            self.action_space = spaces.Box(
                low=np.array(
                    [0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                    dtype=np.float32,
                ),
                high=np.ones(9, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
            # low=np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0]),
            low=np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32
            )
        
        self.max_steps = max_steps
        self.current_step = 0
        self.metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
        
        # Tracking para debugging
        self.episode_stats = {
            'total_reward': 0.0,
            'min_altitude': float('inf'),
            'final_altitude': None,
            'max_throttle': 0.0,
            'engines_used': set()
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            # Rocket.create_random_state() usa random.uniform/choice además
            # de NumPy; ambos deben recibir la misma semilla.
            random.seed(seed)
        state_dict = self.rocket_env.reset()
        self.current_step = 0
        
        # Reset tracking
        self.episode_stats = {
            'total_reward': 0.0,
            'min_altitude': float('inf'),
            'final_altitude': None,
            'max_throttle': 0.0,
            'engines_used': set()
        }
        
        return np.array(state_dict, dtype=np.float32), {}
    
    def step(self, action):
        self.current_step += 1
        rocket_action = self._convert_action(action)
        next_state, reward, done, info = self.rocket_env.step(rocket_action)
        
        # Tracking para debugging
        self.episode_stats['total_reward'] += reward
        altitude = self.rocket_env.state['y'] - self.rocket_env.H/2.0
        self.episode_stats['min_altitude'] = min(self.episode_stats['min_altitude'], altitude)
        
        # Track engine usage
        for i, thrust in enumerate(self.rocket_env.engines_thrust):
            if thrust > 0:
                self.episode_stats['engines_used'].add(i)
                self.episode_stats['max_throttle'] = max(
                    self.episode_stats['max_throttle'],
                    thrust / self.rocket_env.engine_thrust_sl
                )
        
        reached_time_limit = self.current_step >= self.max_steps
        if self.version == "v2" and reached_time_limit:
            # En V2 agotar los 750 pasos es un fallo de la tarea, no una
            # truncacion externa. Marcarlo como terminal evita que SAC haga
            # bootstrap mas alla del timeout pese a haber recibido su coste.
            terminated = True
            truncated = False
        else:
            terminated = bool(done)
            truncated = bool(reached_time_limit and not terminated)
        
        if terminated or truncated:
            self.episode_stats['final_altitude'] = altitude
            # 🔥 NUEVO: Guardar estado final completo para detección precisa de success
            self.episode_stats['final_state'] = {
                'x': self.rocket_env.state['x'],
                'y': self.rocket_env.state['y'],
                'vx': self.rocket_env.state['vx'],
                'vy': self.rocket_env.state['vy'],
                'theta': self.rocket_env.state['theta'],
                'vtheta': self.rocket_env.state['vtheta']
            }
            
            # 🔥 NUEVO: Tracking de estabilización
            required_steps = getattr(self.rocket_env, 'required_stable_steps', 20)
            # Durante el segundo visual posterior al éxito el contador interno
            # sigue avanzando hasta 40, pero el criterio ya está satisfecho.
            self.episode_stats['stable_steps'] = min(
                getattr(self.rocket_env, 'touchdown_stable_steps', 0),
                required_steps,
            )
            self.episode_stats['required_steps'] = required_steps
            self.episode_stats['success'] = bool(self.rocket_env.already_landing)
            if self.episode_stats['success']:
                self.episode_stats['termination_reason'] = 'landing'
            elif reached_time_limit:
                self.episode_stats['termination_reason'] = 'timeout'
            elif self.rocket_env.already_crash:
                self.episode_stats['termination_reason'] = info.get(
                    'termination_reason', 'crash'
                )
            else:
                self.episode_stats['termination_reason'] = 'truncated'

            info['episode_stats'] = self.episode_stats
            # EvalCallback reconoce esta clave y registra success_rate.
            info['is_success'] = self.episode_stats['success']
        
        return np.array(next_state, dtype=np.float32), reward, terminated, truncated, info or {}
    
    # def _convert_action(self, action):
    #     converted = np.zeros(6, dtype=np.float32)
    #     # converted[0:3] = np.clip(action[0:3], 0.0, 1.0)
    #     converted[0:3] = (action[0:3] + 1.0) / 2.0
    #     max_gimbal = 20 * np.pi / 180
    #     converted[3:6] = np.clip(action[3:6], -1.0, 1.0) * max_gimbal
    #     return converted
    
    def _convert_action(self, action):
        """
        Convierte las acciones SAC (9 valores) en comandos físicos del cohete:
        
        ESTRUCTURA:
        action[0:3]  → throttles (rango [0, 1])
        action[3:6]  → gimbals (rango [-1, 1])
        action[6:9]  → on/off (rango [-1, 1])

        V2 mantiene throttle, gimbal y ON/OFF separados. El mapeo lineal
        30-100 % se aplica después en Rocket.step(). Un motor que ya estuvo
        encendido queda bloqueado definitivamente cuando recibe OFF.
        
        MAPEO DE THROTTLES V2:
        - Red produce: [0.0, 1.0]
        - Rocket.step() lo mapea linealmente a: [0.30, 1.0]
        - Fórmula: physical_throttle = 0.30 + 0.70 * network_output
        - El primer step de cada encendido queda limitado al 30 %
        
        MAPEO ON/OFF:
        - on_signal >= +0.20 → encender/continuar si sigue disponible
        - on_signal <= 0.0 → apagar; si estaba ON queda bloqueado
        - 0.0 < on_signal < +0.20 → conservar el estado anterior
        
        APRENDIZAJE:
        - Para encender: Debe aprender on_signal >= +0.20 + throttle [0,1]
        - Para apagar: Debe aprender on_signal <= 0.0 (NO throttle=0)
        - Tras apagar un motor que estaba ON, no puede volver a encenderlo
        """
        if self.engine_mode == 'single':
            # SAC starts near the centre of a Box action and samples uniformly
            # during learning_starts. A linear mapping therefore commands about
            # 50 % thrust, more than twice V1's hover throttle. The cubic curve
            # keeps the complete 0-100 % authority but makes random exploration
            # average 25 % and maps the neutral 0.5 command to 12.5 %.
            throttle_command = float(np.clip(action[0], 0.0, 1.0))
            throttle = throttle_command ** 3
            gimbal = float(np.clip(action[1], -1.0, 1.0)) * np.deg2rad(30.0)
            return np.array([throttle, gimbal], dtype=np.float32)

        if self.engine_mode == 'three_v2':
            converted = np.zeros(9, dtype=np.float32)
            converted[0:3] = np.clip(action[0:3], 0.0, 1.0)
            # Keep V2 gimbals normalized here. Rocket.step() owns the single
            # normalized-command -> physical-angle conversion.
            converted[3:6] = np.clip(action[3:6], -1.0, 1.0)
            converted[6:9] = np.clip(action[6:9], -1.0, 1.0)
            return converted

        converted = np.zeros(9, dtype=np.float32)
        
        # 1. THROTTLES: Mapeo lineal [0,1] → [0.4, 1.0]
        #    - 0.0 → 40% (mínimo operacional Raptor)
        #    - 1.0 → 100% (empuje máximo)
        throttle_raw = np.clip(action[0:3], 0.0, 1.0)
        converted[0:3] = 0.4 + 0.6 * throttle_raw
        
        # 2. GIMBALS: Mantener [-1, 1] (rocket.py convierte a ±20°)
        converted[3:6] = np.clip(action[3:6], -1.0, 1.0)
        
        # 3. ON/OFF: Pasar sin modificar (rocket.py evalúa signo)
        #    - Positivo: motor disponible
        #    - Negativo/Cero: motor apagado (empuje forzado a 0)
        converted[6:9] = action[6:9]
        
        return converted
        
    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode
        if mode == 'rgb_array':
            return self.rocket_env.render(
                return_rgb_array=True,
                render_layout=self.render_layout,
            )
        else:
            return self.rocket_env.render(
                return_rgb_array=False,
                render_layout=self.render_layout,
            )
    
    def close(self):
        if hasattr(self.rocket_env, 'close'):
            self.rocket_env.close()


# ═══════════════════════════════════════════════════════════════
# CALLBACKS OPTIMIZADOS PARA SAC
# ═══════════════════════════════════════════════════════════════
class PolicyResetCallback(BaseCallback):
    """Detecta estancamiento del aprendizaje y resetea parcialmente la política SAC"""
    def __init__(self, check_freq=50000, patience=3, threshold=100, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.threshold = threshold
        self.last_mean_reward = None
        self.stuck_counter = 0

    def _on_step(self) -> bool:
        # Ejecutar cada cierto número de steps (con tolerancia)
        if self.num_timesteps % self.check_freq < 10 and self.num_timesteps > 0:
            try:
                # Obtener el último valor de ep_rew_mean
                logs = self.logger.get_log_dict()
                current_reward = logs.get('rollout/ep_rew_mean', None)
            except Exception:
                current_reward = None

            if current_reward is None:
                return True

            if self.last_mean_reward is None:
                self.last_mean_reward = current_reward
                return True

            # Detectar plateau
            if abs(current_reward - self.last_mean_reward) < self.threshold:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

            self.last_mean_reward = current_reward

            if self.stuck_counter >= self.patience:
                if self.verbose:
                    print(f"\n⚠️ POLÍTICA COLAPSADA - Reinicio parcial ({self.num_timesteps} steps)")

                # Re-inicializar optimizador y añadir ruido a los pesos
                with torch.no_grad():
                    for param in self.model.policy.parameters():
                        param.add_(torch.randn_like(param) * 0.01)

                self.model.policy.optimizer = type(self.model.policy.optimizer)(
                    self.model.policy.parameters(),
                    lr=float(self.model.learning_rate(1.0))
                )

                # Aumentar entropía temporalmente
                if hasattr(self.model, "ent_coef"):
                    self.model.ent_coef = "auto_0.5"

                self.stuck_counter = 0

        return True
    
class SACProgressCallback(BaseCallback):
    """Callback para monitorear y ajustar SAC durante entrenamiento"""
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.landing_successes = []
        self.min_altitudes = []
        self.max_throttles = []
        self.termination_reasons = []
        
    def _on_step(self) -> bool:
        # Recopilar información de episodios terminados
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if "episode_stats" in info:
                stats = info["episode_stats"]
                self.episode_rewards.append(stats['total_reward'])
                self.min_altitudes.append(stats['min_altitude'])
                self.max_throttles.append(stats.get('max_throttle', 0.0))
                self.termination_reasons.append(
                    stats.get('termination_reason', 'unknown')
                )
                
                # Usar la decisión terminal del entorno, no una reconstrucción
                # aproximada a partir del estado final.
                is_success = bool(stats.get('success', False))
                self.landing_successes.append(is_success)

                # Métricas visibles en TensorBoard. La primera es binaria por
                # episodio; la segunda suaviza la curva con una ventana de 50.
                self.logger.record("rollout/landing_success", float(is_success))
                self.logger.record(
                    "rollout/landing_success_rate_50",
                    float(np.mean(self.landing_successes[-50:])),
                )
                self.logger.record(
                    "rollout/landing_stable_steps",
                    float(stats.get('stable_steps', 0)),
                )
                
                # Log cada 100 episodios
                if len(self.episode_rewards) % 50 == 0:
                    recent_rewards = self.episode_rewards[-50:]
                    recent_successes = self.landing_successes[-50:]
                    recent_altitudes = self.min_altitudes[-50:]
                    recent_throttles = self.max_throttles[-50:]
                    recent_reasons = self.termination_reasons[-50:]
                    
                    if self.verbose:
                        print(f"\n[SAC Progress] Episodes: {len(self.episode_rewards)}")
                        print(f"  Avg Reward: {np.mean(recent_rewards):.0f}")
                        print(f"  Success Rate: {np.mean(recent_successes)*100:.1f}%")
                        print(f"  Avg Min Altitude: {np.mean(recent_altitudes):.1f}m")
                        print(f"  Engines Used: {stats.get('engines_used', 'N/A')}")
                        print(f"  Avg Episode Max Throttle: {np.mean(recent_throttles)*100:.1f}%")
                        reason_counts = {
                            reason: recent_reasons.count(reason)
                            for reason in sorted(set(recent_reasons))
                        }
                        print(f"  Terminations: {reason_counts}")

                        # 🔥 NUEVO: Info de estabilización
                        if 'stable_steps' in stats:
                            stable_steps = stats['stable_steps']
                            required = stats.get('required_steps', 20)
                            print(f"  Stability: {stable_steps}/{required} steps ({stable_steps/required*100:.0f}%)")
        
        # Ajuste dinámico de temperatura (ent_coef)
        if hasattr(self.model, 'ent_coef') and self.n_calls % 10000 == 0:
            # Si el agente no está explorando suficiente (hovering)
            if len(self.min_altitudes) > 100:
                recent_min_alt = np.mean(self.min_altitudes[-100:])
                if recent_min_alt > 200:  # Está hovereando muy alto
                    # Aumentar exploración temporalmente
                    if hasattr(self.model.ent_coef, 'set_alpha'):
                        current_alpha = self.model.ent_coef.get_alpha()
                        new_alpha = min(current_alpha * 1.5, 0.5)
                        self.model.ent_coef.set_alpha(new_alpha)
                        if self.verbose:
                            print(f"[SAC] Aumentando exploración: alpha {current_alpha:.4f} → {new_alpha:.4f}")
        
        return True


# ═══════════════════════════════════════════════════════════════
# MODELO SAC OPTIMIZADO
# ═══════════════════════════════════════════════════════════════

def create_optimized_sac_model(
    env,
    log_path="./logs/",
    version=DEFAULT_VERSION,
    curriculum_phase=DEFAULT_CURRICULUM_PHASE,
):
    # policy_kwargs = dict(
    #     net_arch=dict(pi=[768, 768, 384], qf=[768, 768, 384]),
    #     activation_fn=torch.nn.ReLU,
    #     log_std_init=-2.5,
    #     n_critics=2,
    # )

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128],
                      qf=[256, 256, 256]),
        activation_fn=torch.nn.ReLU,
        n_critics=2,
    )

    # SB3 sums log-probabilities over all nine V2 outputs, including masked
    # actuators. Use the same automatic entropy target across V2 phases.
    is_v2 = version == "v2"
    if is_v2:
        ent_coef = "auto_0.01"
        target_entropy = -9.0
    else:
        ent_coef = "auto_0.1"
        target_entropy = "auto"

    learning_rate = 3e-4 if is_v2 else 3e-4
    gamma = 0.999 if is_v2 else 0.998
    gradient_steps = 4

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=600_000,       # Amplio pero no excesivo
        learning_starts=30_000,      # Aprender pronto
        batch_size=1024,             # Buen tamaño para GPU
        gamma=gamma,
        tau=0.005,                   # Target más suave (stabiliza critic)
        train_freq=(1, "step"),                # 🔥 Clave: mantiene ritmo estable
        gradient_steps=gradient_steps,
        target_update_interval=1,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        use_sde=False,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_path,
        device="auto",
        verbose=0,
    )

    return model


# ═══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════

def train_starship_sac(load=False, seed=42,
                       curriculum_phase=DEFAULT_CURRICULUM_PHASE,
                       version=DEFAULT_VERSION,
                       render_layout='default'):
    """
    Entrena Starship con SAC optimizado para aterrizaje
    """
    # Configuración
    task = 'landing'
    version, version_config = resolve_version(version)
    output_name = version_config['output_name']
    if version == "v2":
        output_name = f"{output_name}_{curriculum_phase}"
    total_timesteps = 4_000_000
    n_envs = 8
    model_save_path = f"./models/{output_name}"
    log_path = f"./logs/{output_name}/"
    video_path = f"./videos/{output_name}"
    set_reproducible_seed(seed)
    
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   STARSHIP LANDING - SAC OPTIMIZADO          ║")
    print(f"╚══════════════════════════════════════════════╝")
    print(f"📊 Configuración:")
    print(f"   • Versión: {version} ({version_config['description']})")
    print(f"   • Observaciones: {version_config['observation_dims']}")
    print(f"   • Entornos paralelos: {n_envs}")
    print(f"   • Curriculum: {curriculum_phase}")
    print(f"   • Total timesteps: {total_timesteps:,}")
    print(f"   • Buffer size: 400_000")
    print(f"   • Batch size: 1024")
    print(f"   • Learning starts: 20k")
    if version == "v2":
        print("   • Learning rate: 1e-4")
        print("   • Gamma: 0.999")
        print("   • Gradient steps: 4")
        print("   • Observation normalization: VecNormalize")
    else:
        print("   • Learning rate: 3e-4")
        print("   • Gamma: 0.998")
        print("   • Gradient steps: 4")
        print("   • Observation normalization: VecNormalize")
    if version == "v2":
        entropy_description = "auto_0.01, target -9.0"
    else:
        entropy_description = "auto_0.1"
    print(f"   • Entropy: {entropy_description}")
    print()
    
    # Crear directorios
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)

    checkpoint_store = CheckpointStore(model_save_path, version, curriculum_phase, seed)
    log_path = os.path.join(log_path, checkpoint_store.path.name)
    os.makedirs(log_path, exist_ok=True)
    print(f"[Logs] TensorBoard and evaluations: {log_path}")
    
    # Factory para entornos
    def env_factory():
        return StarshipGymEnv(task=task, max_steps=750, version=version,
                              curriculum_phase=curriculum_phase,
                              render_layout=render_layout)
    
    # Crear entornos vectorizados
    print("🚀 Creando entornos...")
    env = make_vec_env(env_factory, n_envs=n_envs, seed=seed,
                       vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv)
    
    # Compact V2 restores observation normalization in both environments.
    # Evaluation freezes its statistics and receives the training statistics
    # through sync_envs_normalization; reward normalization remains disabled.
    normalize_observations = True
    normalization_gamma = 0.995 if version == "v2" else 0.998
    env = VecNormalize(
        env,
        norm_obs=normalize_observations,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,  # Clip más alto para rewards grandes
        gamma=normalization_gamma
    )
    
    # Crear entorno de evaluación
    print("📹 Configurando evaluación...")
    eval_env = DummyVecEnv([lambda: Monitor(
        StarshipGymEnv(task=task, max_steps=750, render_mode='rgb_array', version=version,
                       curriculum_phase=curriculum_phase,
                       render_layout=render_layout)
    )])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=normalize_observations,
        norm_reward=False,  # No normalizar rewards en eval
        clip_obs=10.0,
        clip_reward=10.0,  # Clip más alto para rewards grandes
        gamma=normalization_gamma,
        training=False
    )
    
    # Crear modelo
    print("🤖 Creando modelo SAC optimizado...")
    model = create_optimized_sac_model(
        env,
        log_path,
        version=version,
        curriculum_phase=curriculum_phase,
    )
    
    # ═══════════════════════════════════════════════════════════════
    # CALLBACKS
    # ═══════════════════════════════════════════════════════════════
    
    print("⚙️ Configurando callbacks...")
    
    # 1. Evaluación
    eval_callback = LandingEvalCallback(
        eval_env,
        store=checkpoint_store,
        eval_seed=seed + 100_000,
        log_path=log_path,
        eval_freq=max(25_000 // n_envs, 1),  # Más frecuente
        n_eval_episodes=30,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # 2. Progress monitoring
    progress_callback = SACProgressCallback(verbose=1)
    
    # 4. Video recording
    video_callback = VideoRecordingCallback(
        eval_env=eval_env,
        video_folder=video_path,
        video_frequency=30_000,
        video_length=1000,
        verbose=1
    )
    

    # Combinar callbacks
    all_callbacks = [
        eval_callback,
        PeriodicCheckpointCallback(checkpoint_store),
        progress_callback,
        # curriculum_callback,
        video_callback,
    ]
    
    # ═══════════════════════════════════════════════════════════════
    # ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "="*50)
    print("🚀 INICIANDO ENTRENAMIENTO SAC")
    print("="*50)
    print("📊 Monitoreo: tensorboard --logdir=./logs/")
    print(f"📹 Videos guardados en: {video_path}/")
    print(f"💾 Modelos guardados en: {model_save_path}/")
    print("="*50 + "\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=all_callbacks,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name="starship_sac_optimized"
        )
        
        # Guardar modelo final
        print("\n💾 Guardando modelo final...")
        checkpoint_store.save(model, 'final', replay=True)
        
        print("✅ Entrenamiento completado exitosamente!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por usuario")
        print("💾 Guardando checkpoint...")
        checkpoint_store.save(model, 'interrupted', replay=True)
        
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        raise
    
    return model, env


# ═══════════════════════════════════════════════════════════════
# CLASE PARA VIDEO RECORDING
# ═══════════════════════════════════════════════════════════════

class VideoRecordingCallback(BaseCallback):
    """Callback para grabación de videos"""
    def __init__(self, eval_env, video_folder="./videos", video_frequency=100000, 
                 video_length=1000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_folder = video_folder
        self.video_frequency = video_frequency
        self.video_length = video_length
        self._last_video_step = 0
        
        os.makedirs(self.video_folder, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_video_step >= self.video_frequency:
            self._record_video()
            self._last_video_step = self.num_timesteps
        return True
    
    def _record_video(self):
        """Graba un video del desempeño actual"""
        try:
            # 🔥 FIX: Sincronizar VecNormalize ANTES de grabar
            training_env = self.model.get_env()
            if isinstance(self.eval_env, VecNormalize) and isinstance(training_env, VecNormalize):
                # La utilidad oficial comprueba si realmente existe obs_rms.
                # Keep the guard for checkpoints that disabled normalization.
                sync_envs_normalization(training_env, self.eval_env)

                if self.verbose and training_env.norm_obs:
                    mean_x = self.eval_env.obs_rms.mean[0]
                    mean_y = self.eval_env.obs_rms.mean[1]
                    print(f"[Video] Sincronizado VecNormalize (x̄={mean_x:.2f}, ȳ={mean_y:.2f})")
                elif self.verbose:
                    print("[Video] V2 usa escalado fisico fijo; no requiere obs_rms")
            
            video_name = f"starship_sac_{self.num_timesteps:09d}"
            recorder = VecVideoRecorder(
                self.eval_env,
                video_folder=self.video_folder,
                record_video_trigger=lambda x: x == 0,
                video_length=self.video_length,
                name_prefix=video_name
            )
            
            obs = recorder.reset()
            for step in range(self.video_length):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = recorder.step(action)
                if np.any(dones):
                    break
            
            recorder.close()
            
            if self.verbose:
                print(f"[Video] ✅ Guardado: {video_name}.mp4 (step {self.num_timesteps:,})")
                
        except Exception as e:
            if self.verbose:
                print(f"[Video] Error: {e}")

# ═══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

   
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", metavar="MODEL.zip",
                      help="Continuar entrenamiento desde un checkpoint SAC")
    mode.add_argument("--evaluate", metavar="MODEL.zip",
                      help="Evaluar un checkpoint sin entrenarlo")
    parser.add_argument("--vecnorm", metavar="VECNORMALIZE.pkl",
                        help="Normalizador asociado al checkpoint (obligatorio para reanudar/evaluar)")
    parser.add_argument(
        "--replay-buffer",
        metavar="REPLAY_BUFFER.pkl",
        help=(
            "Replay buffer SAC asociado al checkpoint (opcional con --resume). "
            "Requiere el mismo contrato de entorno y número de entornos."
        ),
    )
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Pasos adicionales al usar --resume (por defecto: 300000)")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Entornos paralelos al usar --resume (por defecto: 8)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Episodios deterministas al usar --evaluate (por defecto: 10)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate de fine-tuning al usar --resume (por defecto: 1e-4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla base para entornos, NumPy, Python y Torch (por defecto: 42)")
    parser.add_argument(
        "--version",
        choices=tuple(VERSION_CONFIGS),
        default=DEFAULT_VERSION,
        help=f"Versión del entorno (por defecto: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--phase",
        choices=tuple(CURRICULUM_PHASES),
        default=None,
        help=(
            "Fase del curriculum. Por defecto usa v2_phase_1a para V2 y "
            f"{DEFAULT_CURRICULUM_PHASE} para las versiones V1."
        ),
    )
    parser.add_argument(
        "--activation-visualization",
        action="store_true",
        help="Generar GIF del actor y MP4 combinado durante --evaluate",
    )
    parser.add_argument(
        "--render-layout",
        choices=("default", "close_pad"),
        default="default",
        help=(
            "Composicion de video: default conserva el render existente; "
            "close_pad usa camara cercana y proyeccion cenital del pad."
        ),
    )
    args = parser.parse_args()
    selected_version = VERSION_ALIASES.get(args.version, args.version)
    if args.phase is None:
        args.phase = (
            "v2_phase_1a"
            if selected_version == "v2"
            else DEFAULT_CURRICULUM_PHASE
        )
    if args.activation_visualization and not args.evaluate:
        parser.error("--activation-visualization solo se puede usar con --evaluate")
    if args.replay_buffer and not args.resume:
        parser.error("--replay-buffer solo se puede usar junto con --resume")
    
    if args.resume:
        if not args.vecnorm:
            parser.error("--resume requiere --vecnorm para preservar la normalización de observaciones")
        if not os.path.isfile(args.resume):
            parser.error(f"No existe el checkpoint: {args.resume}")
        if not os.path.isfile(args.vecnorm):
            parser.error(f"No existe el VecNormalize: {args.vecnorm}")
        if args.replay_buffer and not os.path.isfile(args.replay_buffer):
            parser.error(f"No existe el replay buffer: {args.replay_buffer}")

        print(f"📂 Cargando modelo desde: {args.resume}")
        set_reproducible_seed(args.seed)

        # La versión debe coincidir con el contrato de observaciones del modelo.
        task = 'landing'
        version, version_config = resolve_version(args.version)
        output_name = version_config['output_name']
        if version == "v2":
            output_name = f"{output_name}_{args.phase}"
        total_timesteps = args.timesteps
        n_envs = args.n_envs
        model_save_path = f"./models/{output_name}_resumed"
        log_path = f"./logs/{output_name}_resumed/"
        resumed_video_path = f"./videos/{output_name}_resumed"
        
        print(f"╔══════════════════════════════════════════════╗")
        print(f"║   REENTRENANDO STARSHIP - SAC OPTIMIZADO     ║")
        print(f"╚══════════════════════════════════════════════╝")
        print(f"📊 Configuración:")
        print(f"   • Versión: {version} ({version_config['description']})")
        print(f"   • Entornos paralelos: {n_envs}")
        print(f"   • Curriculum: {args.phase}")
        print(f"   • Total timesteps: {total_timesteps:,}")
        print(f"   • Cargando desde: {args.resume}")
        print()
        
        # Crear directorios
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(resumed_video_path, exist_ok=True)

        checkpoint_store = CheckpointStore(
            model_save_path, version, args.phase, args.seed, source=args.resume
        )
        log_path = os.path.join(log_path, checkpoint_store.path.name)
        os.makedirs(log_path, exist_ok=True)
        print(f"[Logs] TensorBoard and evaluations: {log_path}")
        
        # Factory para entornos
        def env_factory():
            return StarshipGymEnv(task=task, max_steps=750, version=version,
                                  curriculum_phase=args.phase,
                                  render_layout=args.render_layout)
        
        # Crear entornos vectorizados
        print("🚀 Creando entornos...")
        env = make_vec_env(
            env_factory,
            n_envs=n_envs,
            seed=args.seed,
            vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
        )
        
        # Cargar VecNormalize si existe
        vecnorm_path = args.vecnorm
        print(f"✅ Cargando VecNormalize desde: {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = False
        
        # Crear entorno de evaluación
        print("📹 Configurando evaluación...")
        eval_env = DummyVecEnv([lambda: Monitor(
            StarshipGymEnv(task=task, max_steps=750, render_mode='rgb_array', version=version,
                           curriculum_phase=args.phase,
                           render_layout=args.render_layout)
        )])
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        
        # Cargar modelo
        print(f"🤖 Cargando modelo SAC...")
        # Without replay, retain the update warm-up but collect with the
        # loaded policy instead of sampling uniform random ON/OFF commands.
        resume_model_class = SAC if args.replay_buffer else PolicyWarmupSAC
        model = resume_model_class.load(args.resume, env=env, device="auto")
        # Loaded models retain their original TensorBoard directory.
        # Route this continuation to its own phase/session before learn().
        model.tensorboard_log = log_path
        model.set_random_seed(args.seed)
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        for optimizer in (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer):
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.learning_rate
        if args.replay_buffer:
            print(f"🧠 Cargando replay buffer desde: {args.replay_buffer}")
            model.load_replay_buffer(args.replay_buffer)

            replay_buffer = model.replay_buffer
            buffer_n_envs = int(replay_buffer.n_envs)
            if buffer_n_envs != n_envs:
                raise ValueError(
                    "El replay buffer fue creado con "
                    f"{buffer_n_envs} entornos, pero --n-envs={n_envs}. "
                    f"Repite el comando con --n-envs {buffer_n_envs}."
                )

            expected_obs_shape = tuple(env.observation_space.shape)
            buffer_obs_shape = tuple(replay_buffer.obs_shape)
            if buffer_obs_shape != expected_obs_shape:
                raise ValueError(
                    "Observaciones incompatibles entre replay buffer y entorno: "
                    f"{buffer_obs_shape} != {expected_obs_shape}."
                )

            expected_action_dim = int(np.prod(env.action_space.shape))
            if int(replay_buffer.action_dim) != expected_action_dim:
                raise ValueError(
                    "Acciones incompatibles entre replay buffer y entorno: "
                    f"{replay_buffer.action_dim} != {expected_action_dim}."
                )

            # num_timesteps ya supera normalmente learning_starts. Dejarlo a
            # cero hace explícito que un buffer restaurado no necesita warm-up.
            model.learning_starts = 0
            stored_transitions = (
                replay_buffer.buffer_size * replay_buffer.n_envs
                if replay_buffer.full
                else replay_buffer.pos * replay_buffer.n_envs
            )
            print(
                "✅ Replay buffer restaurado: "
                f"{stored_transitions:,} transiciones; entrenamiento inmediato"
            )
        else:
            # SAC.save() no incluye el replay buffer. Recoger experiencias con
            # la política cargada antes de actualizar. PolicyWarmupSAC separa
            # este umbral de aprendizaje del muestreo de acciones: nunca usa
            # action_space.sample() durante el warm-up del resume.
            model.learning_starts = model.num_timesteps + 50_000
            print(
                "🧠 Sin replay buffer: recogida de 50,000 transiciones con "
                "la política cargada (sin acciones uniformes aleatorias). "
                "Las redes se actualizarán después de esta recogida."
            )
        print(f"🔢 Timesteps previos: {model.num_timesteps:,}")
        
        # Configurar callbacks (IDÉNTICOS al original)
        print("⚙️ Configurando callbacks...")
        
        eval_callback = LandingEvalCallback(
            eval_env,
            store=checkpoint_store,
            eval_seed=args.seed + 100_000,
            log_path=log_path,
            eval_freq=max(25_000 // n_envs, 1),
            n_eval_episodes=30,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        progress_callback = SACProgressCallback(verbose=1)
        
        video_callback = VideoRecordingCallback(
            eval_env=eval_env,
            video_folder=resumed_video_path,
            video_frequency=30_000,
            video_length=1000,
            verbose=1
        )
        
        all_callbacks = [
            eval_callback,
            PeriodicCheckpointCallback(checkpoint_store),
            progress_callback,
            video_callback,
        ]
        
        # Entrenamiento
        print("\n" + "="*50)
        print("🚀 REANUDANDO ENTRENAMIENTO SAC")
        print("="*50)
        print("📊 Monitoreo: tensorboard --logdir=./logs/")
        print(f"📹 Videos guardados en: {resumed_video_path}/")
        print(f"💾 Modelos guardados en: {model_save_path}/")
        print("="*50 + "\n")
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=all_callbacks,
                progress_bar=True,
                reset_num_timesteps=False,
                tb_log_name="starship_sac_resumed"
            )
            
            # Guardar modelo final
            print("\n💾 Guardando modelo final...")
            checkpoint_store.save(model, 'final', replay=True)
            
            print("✅ Reentrenamiento completado exitosamente!")
            
        except KeyboardInterrupt:
            print("\n⚠️ Entrenamiento interrumpido por usuario")
            print("💾 Guardando checkpoint...")
            checkpoint_store.save(model, 'interrupted', replay=True)
            
        except Exception as e:
            print(f"\n❌ Error durante reentrenamiento: {e}")
            raise

    elif args.evaluate:
        import imageio

        if not args.vecnorm:
            parser.error("--evaluate requiere --vecnorm para usar la misma normalización del entrenamiento")
        if not os.path.isfile(args.evaluate):
            parser.error(f"No existe el checkpoint: {args.evaluate}")
        if not os.path.isfile(args.vecnorm):
            parser.error(f"No existe el VecNormalize: {args.vecnorm}")
        set_reproducible_seed(args.seed)
        version, version_config = resolve_version(args.version)
        print(f"🧪 Versión de evaluación: {version} ({version_config['description']})")
        print(f"🎓 Curriculum de evaluación: {args.phase}")
        
        print("\n╔════════════════════════════════════════════════╗")
        print("║   EVALUACIÓN COMPLETA - 10 EPISODIOS           ║")
        print("╚════════════════════════════════════════════════╝\n")

        # === Rutas ===
        model_path = args.evaluate
        vecnorm_path = args.vecnorm
        video_folder = f"./videos/evaluation/{version}/{args.phase}/"
        os.makedirs(video_folder, exist_ok=True)

        # === Crear entorno ===
        print("🔄 Creando entorno de evaluación...")
        def make_eval_env():
            return Monitor(StarshipGymEnv(
                task="landing",
                max_steps=750,
                render_mode="rgb_array",
                version=version,
                curriculum_phase=args.phase,
                render_layout=args.render_layout,
            ))

        eval_env = DummyVecEnv([make_eval_env])
        print(f"✅ Cargando normalizador desde: {vecnorm_path}")
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        # === Cargar modelo ===
        print(f"📦 Cargando modelo desde: {model_path}")
        model = SAC.load(model_path, env=eval_env, device="auto")

        activation_recorder = None
        export_activation_visualization = None
        if args.activation_visualization:
            from activation_visualizer import (
                ActorActivationRecorder,
                export_activation_visualization,
            )

            activation_recorder = ActorActivationRecorder(model.actor)
            print("🧠 Visualización de activaciones del actor habilitada")

        # === Evaluación de 10 episodios ===
        n_eval_episodes = args.episodes
        results = {
            'success': 0,
            'crash': 0,
            'timeout': 0,
            'rewards': [],
            'final_altitudes': [],
            'final_velocities': [],
            'landing_speeds': []
        }

        print(f"\n🚀 Ejecutando {n_eval_episodes} episodios de evaluación...\n")
        print("="*60)

        for ep in range(n_eval_episodes):
            # Cada episodio tiene una semilla conocida e independiente: el
            # mismo comando genera los mismos escenarios y mismos vídeos.
            eval_env.seed(args.seed + ep)
            obs = eval_env.reset()
            frames = []
            activation_samples = []
            episode_reward = 0
            done = [False]
            step_count = 0

            while not done[0]:
                if activation_recorder is not None:
                    activation_recorder.begin_step()
                action, _ = model.predict(obs, deterministic=True)
                if activation_recorder is not None:
                    activation_samples.append(activation_recorder.capture(
                        observation=obs[0],
                        action=action[0],
                        step=step_count,
                    ))
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                step_count += 1

                # Capturar frame
                frame = eval_env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)

                if done[0]:
                    # Extraer estado final
                    episode_stats = info[0].get("episode_stats", {})
                    state = episode_stats.get("final_state", {})
                    # final_state['y'] es la coordenada del centro de masas
                    # (25 m al tocar). El wrapper ya guarda AGL correctamente.
                    final_altitude = episode_stats.get("final_altitude", 0.0)
                    final_vy = abs(state.get("vy", 0))
                    final_vx = abs(state.get("vx", 0))
                    final_speed = np.sqrt(final_vx**2 + final_vy**2)
                    
                    results['final_altitudes'].append(final_altitude)
                    results['final_velocities'].append(final_speed)
                    results['rewards'].append(episode_reward)

                    # Clasificar resultado
                    if episode_stats.get("success", False):
                        results['success'] += 1
                        results['landing_speeds'].append(final_vy)
                        status = "✅ ATERRIZAJE EXITOSO"
                    elif step_count >= 750:
                        results['timeout'] += 1
                        status = "⏱️ TIMEOUT"
                    else:
                        results['crash'] += 1
                        status = "💥 CRASH"

                    # Mostrar resultado del episodio
                    print(f"Episodio {ep+1:2d} | {status}")
                    print(f"  Reward: {episode_reward:8.1f} | Alt: {final_altitude:5.1f}m | "
                          f"Vel: {final_speed:5.2f}m/s | Steps: {step_count}")
                    print("-"*60)
                    break

            # Guardar video
            if frames:
                video_path = os.path.join(video_folder, f"episode_{ep+1:02d}.mp4")
                imageio.mimsave(video_path, frames, fps=30)
                if activation_recorder is not None:
                    try:
                        activation_paths = export_activation_visualization(
                            simulation_frames=frames,
                            samples=activation_samples,
                            actor=model.actor,
                            output_folder=video_folder,
                            episode_number=ep + 1,
                            phase=args.phase,
                            status=status,
                        )
                        print(f"  🧠 GIF de red: {activation_paths['network_gif']}")
                        print(f"  🎬 Vídeo combinado: {activation_paths['combined_video']}")
                    except Exception as activation_error:
                        print(f"  ⚠️ No se pudo exportar la visualización: {activation_error}")

        if activation_recorder is not None:
            activation_recorder.close()

        # === RESUMEN FINAL ===
        print("\n" + "="*60)
        print("📊 RESUMEN DE EVALUACIÓN")
        print("="*60)
        print(f"\n🎯 RESULTADOS:")
        print(f"  ✅ Aterrizajes exitosos: {results['success']}/{n_eval_episodes} ({results['success']/n_eval_episodes*100:.1f}%)")
        print(f"  💥 Crashes:              {results['crash']}/{n_eval_episodes} ({results['crash']/n_eval_episodes*100:.1f}%)")
        print(f"  ⏱️  Timeouts:             {results['timeout']}/{n_eval_episodes} ({results['timeout']/n_eval_episodes*100:.1f}%)")
        
        print(f"\n📈 ESTADÍSTICAS DE RECOMPENSA:")
        print(f"  Media:                  {np.mean(results['rewards']):8.1f}")
        print(f"  Desv. estándar:         {np.std(results['rewards']):8.1f}")
        print(f"  Máxima:                 {np.max(results['rewards']):8.1f}")
        print(f"  Mínima:                 {np.min(results['rewards']):8.1f}")
        
        if results['landing_speeds']:
            print(f"\n🛬 VELOCIDADES DE ATERRIZAJE (solo exitosos):")
            print(f"  Media:                  {np.mean(results['landing_speeds']):5.2f} m/s")
            print(f"  Mínima:                 {np.min(results['landing_speeds']):5.2f} m/s")
            print(f"  Máxima:                 {np.max(results['landing_speeds']):5.2f} m/s")
        
        print(f"\n📐 ALTITUDES FINALES:")
        print(f"  Media:                  {np.mean(results['final_altitudes']):6.1f} m")
        print(f"  Mínima:                 {np.min(results['final_altitudes']):6.1f} m")
        print(f"  Máxima:                 {np.max(results['final_altitudes']):6.1f} m")
        
        print(f"\n📹 VIDEOS:")
        print(f"  Ubicación:              {video_folder}")
        print(f"  Total grabados:         {n_eval_episodes} episodios")
        
        print("\n" + "="*60)
        print("✅ Evaluación completada exitosamente")
        print("="*60 + "\n")

    else:
        # Modo entrenamiento normal
        model, env = train_starship_sac(
            seed=args.seed,
            curriculum_phase=args.phase,
            version=args.version,
            render_layout=args.render_layout,
        )
