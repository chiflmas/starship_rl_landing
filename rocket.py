import numpy as np
import random
import cv2
import utils
import math
from curriculum_phases import DEFAULT_CURRICULUM_PHASE, get_curriculum_phase

# class RealisticEngineFlames:
#     """
#     Renderizado realista de llamas de motores Raptor con:
#     - Múltiples colores de llama (azul, blanco, naranja)
#     - Efectos de turbulencia
#     - Variación temporal
#     - Scaling por throttle
#     - Direccionalidad por gimbal
#     """
    
#     def __init__(self):
#         self.frame_count = 0
        
#     def create_realistic_engine_flames(self, rocket, polys):
#         """
#         Crear llamas realistas para motores Starship
#         Reemplaza el método en create_polygons() de rocket.py
#         """
#         if rocket.rocket_type != 'starship':
#             return  # Solo para Starship
            
#         H, W = rocket.H, rocket.H/2.6
#         nozzle_bases = [(-0.12*W, -H/2.0), (0.0, -H/2.0), (0.12*W, -H/2.0)]
        
#         self.frame_count += 1
        
#         for i, (nx, ny) in enumerate(nozzle_bases):
#             thrust_i = rocket.engines_thrust[i] if i < len(rocket.engines_thrust) else 0.0
#             if thrust_i <= 0:
#                 continue
                
#             vphi_i = rocket.engine_gimbals[i] if i < len(rocket.engine_gimbals) else 0.0
#             throttle_rel = np.clip(thrust_i / max(rocket.engine_thrust_sl, 1e-6), 0.0, 1.0)
            
#             # Crear múltiples capas de llamas
#             flame_layers = self._create_flame_layers(
#                 nx, ny, vphi_i, throttle_rel, i
#             )
            
#             # Añadir cada capa a los polígonos
#             for layer in flame_layers:
#                 polys['engine_work'].append(layer)
    
#     def _create_flame_layers(self, nx, ny, gimbal_angle, throttle, engine_id):
#         """Crear múltiples capas de llama para realismo"""
#         layers = []
        
#         # Parámetros base
#         base_length = 150  # longitud base de llama
#         base_width = 20    # ancho base
        
#         # Scaling por throttle
#         length_scale = 0.3 + 1.2 * throttle
#         width_scale = 0.5 + 0.8 * throttle
        
#         # Turbulencia temporal
#         time_factor = (self.frame_count + engine_id * 17) * 0.2
#         turbulence = 0.1 * math.sin(time_factor) + 0.05 * math.sin(time_factor * 2.3)
        
#         # Direcciones por gimbal
#         cos_g, sin_g = math.cos(gimbal_angle), math.sin(gimbal_angle)
        
#         # 1. CORE FLAME (azul brillante - temperatura más alta)
#         core_flame = self._create_flame_polygon(
#             nx, ny, cos_g, sin_g,
#             length=base_length * 0.4 * length_scale,
#             width=base_width * 0.3 * width_scale,
#             turbulence=turbulence * 0.5,
#             color=(255, 255, 200),  # Azul muy claro/blanco
#             segments=8
#         )
#         layers.append(core_flame)
        
#         # 2. INNER FLAME (azul medio)
#         inner_flame = self._create_flame_polygon(
#             nx, ny, cos_g, sin_g,
#             length=base_length * 0.7 * length_scale,
#             width=base_width * 0.6 * width_scale,
#             turbulence=turbulence * 0.8,
#             color=(255, 200, 100),  # Azul medio
#             segments=12
#         )
#         layers.append(inner_flame)
        
#         # 3. OUTER FLAME (azul oscuro/violeta)
#         outer_flame = self._create_flame_polygon(
#             nx, ny, cos_g, sin_g,
#             length=base_length * 1.0 * length_scale,
#             width=base_width * 1.0 * width_scale,
#             turbulence=turbulence * 1.0,
#             color=(180, 120, 60),  # Azul más oscuro
#             segments=16
#         )
#         layers.append(outer_flame)
        
#         # 4. SHOCK DIAMONDS (efectos de onda de choque)
#         if throttle > 0.6:  # Solo con alto throttle
#             diamonds = self._create_shock_diamonds(
#                 nx, ny, cos_g, sin_g, throttle, turbulence
#             )
#             layers.extend(diamonds)
        
#         # 5. PLUME EXPANSION (expansión de gases)
#         if throttle > 0.3:
#             expansion = self._create_plume_expansion(
#                 nx, ny, cos_g, sin_g, throttle, turbulence
#             )
#             layers.extend(expansion)
        
#         return layers
    
#     def _create_flame_polygon(self, nx, ny, cos_g, sin_g, length, width, 
#                             turbulence, color, segments=12):
#         """Crear un polígono de llama con forma realista"""
#         points = []
        
#         # Base de la tobera (ancho máximo)
#         base_width = width * 0.8
#         points.append((-base_width/2, 0))
#         points.append((base_width/2, 0))
        
#         # Puntos a lo largo de la llama
#         for i in range(1, segments):
#             t = i / segments
            
#             # Forma cónica con turbulencia
#             y_pos = -length * t
#             width_at_t = base_width * (1.0 - t * 0.8)  # Se estrecha
            
#             # Añadir turbulencia
#             turb_offset = turbulence * width * math.sin(t * math.pi * 3 + self.frame_count * 0.1)
            
#             # Puntos izquierdo y derecho
#             left_x = -width_at_t/2 + turb_offset
#             right_x = width_at_t/2 + turb_offset
            
#             points.insert(1, (left_x, y_pos))
#             points.append((right_x, y_pos))
        
#         # Punta de la llama
#         tip_turbulence = turbulence * width * 0.5
#         points.insert(len(points)//2, (tip_turbulence, -length))
        
#         # Transformar por gimbal angle
#         transformed_points = []
#         for px, py in points:
#             # Rotar por ángulo de gimbal
#             rx = px * cos_g - py * sin_g
#             ry = px * sin_g + py * cos_g
#             # Trasladar a posición de tobera
#             transformed_points.append((nx + rx, ny + ry))
        
#         return {
#             'pts': transformed_points,
#             'face_color': color,
#             'edge_color': None
#         }
    
#     def _create_shock_diamonds(self, nx, ny, cos_g, sin_g, throttle, turbulence):
#         """Crear efectos de onda de choque (shock diamonds)"""
#         diamonds = []
        
#         # Parámetros
#         diamond_spacing = 30
#         diamond_width = 8
#         diamond_height = 15
        
#         num_diamonds = int(3 * throttle)  # Más diamonds con más throttle
        
#         for i in range(num_diamonds):
#             y_offset = -(diamond_spacing * (i + 1))
            
#             # Crear diamond shape
#             points = [
#                 (0, y_offset),  # top
#                 (diamond_width/2, y_offset - diamond_height/2),  # right
#                 (0, y_offset - diamond_height),  # bottom
#                 (-diamond_width/2, y_offset - diamond_height/2)  # left
#             ]
            
#             # Transformar por gimbal
#             transformed_points = []
#             for px, py in points:
#                 rx = px * cos_g - py * sin_g
#                 ry = px * sin_g + py * cos_g
#                 transformed_points.append((nx + rx, ny + ry))
            
#             # Shock diamonds son muy brillantes
#             diamonds.append({
#                 'pts': transformed_points,
#                 'face_color': (255, 255, 255),  # Blanco brillante
#                 'edge_color': None
#             })
        
#         return diamonds
    
#     def _create_plume_expansion(self, nx, ny, cos_g, sin_g, throttle, turbulence):
#         """Crear expansión del penacho de gases"""
#         expansions = []
        
#         # Crear varias "burbujas" de expansión
#         for i in range(3):
#             expansion_distance = 80 + i * 25
#             expansion_width = 30 + i * 10
            
#             # Posición con turbulencia
#             turb_x = turbulence * 5 * math.sin(self.frame_count * 0.1 + i)
#             turb_y = turbulence * 3 * math.cos(self.frame_count * 0.15 + i)
            
#             # Puntos de la expansión (forma irregular)
#             points = []
#             num_points = 8
#             for j in range(num_points):
#                 angle = 2 * math.pi * j / num_points
#                 radius = expansion_width * (0.7 + 0.3 * math.sin(angle * 2 + self.frame_count * 0.05))
                
#                 px = radius * math.cos(angle) + turb_x
#                 py = -expansion_distance + radius * 0.3 * math.sin(angle) + turb_y
                
#                 # Transformar por gimbal
#                 rx = px * cos_g - py * sin_g
#                 ry = px * sin_g + py * cos_g
#                 points.append((nx + rx, ny + ry))
            
#             # Color más tenue para expansión
#             alpha_factor = max(0.3, 1.0 - i * 0.2)
#             color = (int(150 * alpha_factor), int(100 * alpha_factor), int(50 * alpha_factor))
            
#             expansions.append({
#                 'pts': points,
#                 'face_color': color,
#                 'edge_color': None
#             })
        
#         return expansions

class Rocket(object):
    """
    Rocekt and environment.
    The rocket is simplified into a rigid body model with a thin rod,
    considering acceleration and angular acceleration and air resistance
    proportional to velocity.

    There are two tasks: hover and landing
    Their reward functions are straight forward and simple.

    For the hover tasks: the step-reward is given based on two factors
    1) the distance between the rocket and the predefined target point
    2) the angle of the rocket body (the rocket should stay as upright as possible)

    For the landing task: the step-reward is given based on three factors:
    1) the distance between the rocket and the predefined landing point.
    2) the angle of the rocket body (the rocket should stay as upright as possible)
    3) Speed and angle at the moment of contact with the ground, when the touching-speed
    are smaller than a safe threshold and the angle is close to 90 degrees (upright),
    we see it as a successful landing.

    """

    def __init__(self, max_steps, task='hover', rocket_type='falcon',
                 viewport_h=900, path_to_bg_img=None, engine_mode='three',
                 curriculum_phase=DEFAULT_CURRICULUM_PHASE,
                 observation_mode=None):

        self.task = task
        self.rocket_type = rocket_type
        self.engine_mode = engine_mode
        default_observation_modes = {
            'single': 'v1_engineered',
            'three_v2': 'v2_raw',
            'three': 'legacy',
        }
        self.observation_mode = observation_mode or default_observation_modes.get(
            engine_mode, 'legacy'
        )
        valid_observation_modes = {
            'v1_engineered', 'v1_raw', 'v2_raw', 'legacy'
        }
        if self.observation_mode not in valid_observation_modes:
            raise ValueError(
                f"Unknown observation mode {self.observation_mode!r}. "
                f"Available: {sorted(valid_observation_modes)}"
            )
        valid_mode_contracts = {
            'single': {'v1_engineered', 'v1_raw'},
            'three_v2': {'v2_raw'},
            'three': {'legacy'},
        }
        if self.engine_mode not in valid_mode_contracts:
            raise ValueError(f"Unknown engine mode {self.engine_mode!r}")
        if self.observation_mode not in valid_mode_contracts[self.engine_mode]:
            raise ValueError(
                f"Engine mode {self.engine_mode!r} is incompatible with "
                f"observation mode {self.observation_mode!r}"
            )
        self.curriculum_phase_name = curriculum_phase
        self.curriculum_phase = get_curriculum_phase(curriculum_phase)
        self.debug_mode = False

        self.g = 9.8
        self.H = 50  # rocket height (meters)
        # self.I = 1/12*self.H*self.H  # Moment of inertia
        self.dt = 0.05

        self.world_x_min = -400  # meters
        self.world_x_max = 400
        self.world_y_min = -50
        self.world_y_max = 1000



        if self.rocket_type == 'starship':
            self.mass = 185_000.0 # 140.000 v1
            if self.engine_mode == 'single':
                # V1: actuador virtual centrado equivalente a tres Raptor.
                self.engine_thrust_sl = 6e6
                self.num_engines = 1
                self.min_engine_throttle = 0.0
            elif self.engine_mode == 'three_v2':
                # Three independent sea-level engines. Positive commands are
                # constrained to the explicit 30-100 % operating interval.
                self.engine_thrust_sl = 2.45e6
                self.num_engines = 3
                self.min_engine_throttle = 0.30
                self.engine_switch_on_threshold = 0.20
                self.engine_switch_off_threshold = 0.0
                # Effective 2-D projection of the three-engine cluster. The
                # recovered environment used approximately +/-1.15 m; keeping
                # that spacing reduces differential-thrust torque without
                # unrealistically collapsing all thrust lines to one point.
                self.engine_x_offsets = (-1.15, 0.0, 1.15)
            else:
                self.engine_thrust_sl = 2e6
                self.num_engines = 3
                self.min_engine_throttle = 0.40
            # self.min_stable_throttle = 0.4
        else:
            self.mass = 20_000.0
            self.engine_thrust_sl = 8.5e5
            self.num_engines = 1

        # Estado persistente del motor. Un apagado nominal no es un fallo:
        # - operational: el motor está sano y puede producir empuje.
        # - failed: fallo físico; el motor no puede producir empuje.
        # - locked_out: apagado por la secuencia de landing; no se reenciende
        #   durante este episodio, aunque siga estando sano.
        self.engine_operational = [True] * self.num_engines
        self.engine_failed = [False] * self.num_engines
        self.engine_locked_out = [False] * self.num_engines
        # Inercia transversal. V1 usa un cilindro uniforme de 50 m y diámetro
        # aproximado H/5.5 (el mismo que emplea el área frontal). La versión
        # legacy de tres motores conserva su ajuste original para no alterar
        # entrenamientos ni checkpoints anteriores.
        if self.engine_mode in {'single', 'three_v2'}:
            body_radius = self.H / (2.0 * 5.5)
            self.I = (self.mass / 12.0) * (self.H ** 2 + 3.0 * body_radius ** 2)
        else:
            self.I = (1.0 / 3.0) * self.mass * (self.H ** 2)

        # Estado de motores (una sola vez)
        self.engine_available = [True] * self.num_engines
        self.curriculum_engine_mask = [True] * self.num_engines
        self.engines_on       = [False] * self.num_engines
        self.engines_thrust   = [0.0]   * self.num_engines
        self.engine_throttles = [0.0]   * self.num_engines
        self.engine_gimbals   = [0.0]   * self.num_engines

        if self.task == 'hover':
            self.target_x, self.target_y, self.target_r = 0, 200, 50
        elif self.task == 'landing':
            self.target_x, self.target_y, self.target_r = 0, self.H/2.0, 50

        self.already_landing = False
        self.already_crash = False
        self.touchdown_stable_steps = 0  # Contador de estabilidad
        self.touchdown_start_step = None  # Cuándo tocó suelo
        self.touchdown_contact = False  # El aterrizaje no permite volver a despegar
        self.touchdown_contact_step = None
        self.max_touchdown_settling_steps = 40  # 2 s a 20 Hz para asentarse
        self.required_stable_steps = 20  # 1 segundo a 20Hz
        self.success_display_steps = 20  # Un segundo adicional para mostrar éxito
        self.success_display_start_step = None
        self.success_display_active = False
        self.landing_reward_granted = False
        self.max_steps = max_steps

        # viewport height x width (pixels)
        self.viewport_h = int(viewport_h)
        self.viewport_w = int(viewport_h * (self.world_x_max-self.world_x_min) \
                          / (self.world_y_max - self.world_y_min))
        self.step_id = 0

        self.state = self.create_random_state()
        self.action_table = self.create_action_table()

        # Observation dimensions are explicit so a new version cannot silently
        # load a checkpoint trained with a different input contract.
        observation_dims = {
            'v1_engineered': 12,
            'v1_raw': 10,
            'v2_raw': 17,
            'legacy': 23,
        }
        self.state_dims = observation_dims[self.observation_mode]

        if self.rocket_type == 'starship':
            # Espacio continuo: [t0, t1, t2, v0, v1, v2]
            action_dims = {'single': 2, 'three_v2': 9, 'three': 9}
            self.action_dims = action_dims[self.engine_mode]
            self.continuous_action_space = True
        else:
            # Mantener compatible con Falcon (discreto)
            self.action_dims = len(self.action_table)
            self.continuous_action_space = False


        if path_to_bg_img is None:
            path_to_bg_img = task+'.jpg'
        # Keep the original artwork path for the native close-camera renderer.
        # ``bg_img`` remains the legacy full-world raster used by the default
        # layout, so that layout is bit-for-bit unaffected by the new camera.
        self.path_to_bg_img = path_to_bg_img
        self._close_bg_source = None
        self._close_bg_source_loaded = False
        self.bg_img = utils.load_bg_img(path_to_bg_img, w=self.viewport_w, h=self.viewport_h)

        self.state_buffer = []


        self.angular_damping = 1.5e6       # N·m·s, evita oscilaciones

        # ═════════════════════════════════════════════════════════════════
        # 🔥 NUEVO: Constantes para auto-enderezamiento y vuelco
        # ═════════════════════════════════════════════════════════════════
        
        if self.rocket_type == 'starship':
            # Geometría de la base de aterrizaje (patas de Starship)
            self.base_width = 8.0  # metros (distancia entre patas externas)
            self.base_radius = self.base_width / 2.0  # Radio de estabilidad
            
            # Constantes físicas para enderezamiento
            self.k_restore = 2.5e5   # Torque restaurador (N·m/rad)
            self.k_topple = 1.0e6    # Torque de vuelco (N·m/rad)
            
            # Fricción rotacional adaptativa
            self.c_friction_stable = 8e6   # Alta fricción cuando estable
            self.c_friction_topple = 2e6   # Baja fricción cuando volcando
            
            # Debug opcional
            self.debug_ground_physics = False  # Cambiar a True para ver logs
            
        else:  # falcon u otros
            # Valores por defecto (más conservadores)
            self.base_width = 4.0
            self.base_radius = 2.0
            self.k_restore = 1.0e5
            self.k_topple = 5.0e5
            self.c_friction_stable = 5e6
            self.c_friction_topple = 1e6
            self.debug_ground_physics = False

    def reset(self, state_dict=None):
        """
        Reinicia el episodio:
        - Estado inicial (aleatorio o provisto)
        - Motores: estado inicial definido por la fase; salud, fallo y bloqueo
          de secuencia separados
        """
        # --- Estado inicial ---
        if state_dict is None:
            self.state = self.create_random_state()
        else:
            self.state = state_dict

        # V1/V2 preserve the signed initial belly-flop attitude. The symmetric
        # aerodynamic moment fades before the terminal landing burn.
        # aerodinámico se desvanece al entrar en el landing burn.
        if self.engine_mode in {'single', 'three_v2'}:
            theta0 = float(self.state.get('theta', 0.0))
            theta_sign = np.sign(theta0) if abs(theta0) > 1e-6 else 1.0
            self.belly_flop_theta_reference = theta_sign * np.deg2rad(80.0)

        self.state_buffer = []
        self.step_id = 0
        self.already_landing = False

        # 🔥 NUEVO: Reset de contadores de estabilidad
        self.touchdown_stable_steps = 0
        self.touchdown_start_step = None
        self.touchdown_contact = False
        self.touchdown_contact_step = None
        self.success_display_start_step = None
        self.success_display_active = False
        self.landing_reward_granted = False

        # --- Disponibilidad de motores (fallos al arranque) ---
        if self.rocket_type == 'starship':
            # Sistema de fallos más realista
            self.engine_available, self.engines_failed_at_start = self._generate_engine_failure_config()

        else:
            # Por defecto: todos disponibles
            self.engine_available = [True] * self.num_engines
            self.engines_failed_at_start = []

        # Salud física y bloqueo de secuencia son estados distintos.
        failure_available = list(self.engine_available)
        curriculum_mask = self.curriculum_phase.get('engine_available_mask')
        if self.engine_mode == 'three_v2' and curriculum_mask is not None:
            if len(curriculum_mask) != self.num_engines:
                raise ValueError(
                    "engine_available_mask debe tener un valor por motor"
                )
            curriculum_mask = [bool(value) for value in curriculum_mask]
        else:
            curriculum_mask = [True] * self.num_engines

        self.curriculum_engine_mask = list(curriculum_mask)

        # The curriculum mask is not a physical failure or a permanent
        # shutdown. Engine health keeps its normal V2 semantics; the mask is
        # checked independently by the actuator while this phase is active.
        self.engine_available = list(failure_available)
        self.engine_operational = list(failure_available)
        self.engine_failed = [not available for available in failure_available]
        self.engine_locked_out = [False] * self.num_engines

        # --- Estado instantáneo de propulsión ---
        # Por defecto arrancan apagados; el curriculum puede encender al reset
        # únicamente los actuadores habilitados y operativos.
        initial_on_mask = self.curriculum_phase.get(
            'initial_engine_on_mask',
            [False] * self.num_engines,
        )
        if len(initial_on_mask) != self.num_engines:
            raise ValueError(
                "initial_engine_on_mask debe tener un valor por motor"
            )
        self.engines_on = [
            bool(initial_on)
            and bool(curriculum_enabled)
            and bool(operational)
            for initial_on, curriculum_enabled, operational in zip(
                initial_on_mask,
                self.curriculum_engine_mask,
                self.engine_operational,
            )
        ]
        self.engine_throttles = [
            getattr(self, 'min_engine_throttle', 0.0) if is_on else 0.0
            for is_on in self.engines_on
        ]
        self.engines_thrust = [
            throttle * self.engine_thrust_sl
            for throttle in self.engine_throttles
        ]
        self.engine_gimbals = [0.0] * self.num_engines

        self.last_action = np.zeros(self.action_dims, dtype=np.float32)
        
        # --- Memoria del sistema de recompensas ---
        # El sistema de rewards empieza cada episodio sin memoria previa.
        if hasattr(self, 'reward_system'):
            self.reward_system.reset_episode_memory()


        # --- Limpieza de ventanas (si procede) ---
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

        return self.flatten(self.state)

    def create_action_table(self):
        vphi0 = 0
        vphi1 = 30 / 180 * np.pi
        vphi2 = -30 / 180 * np.pi
        # Solo direcciones de tobera; el empuje lo dará 'throttle' continuo
        return [[None, vphi0], [None, vphi1], [None, vphi2]]

    

    def _clip_gimbal(self, phi):
        # Permitir más rango cuando solo hay motor lateral
        if (
            self.engine_mode == 'three'
            and hasattr(self, 'lateral_active')
            and self.lateral_active
        ):
            # Mayor autoridad para motor lateral (necesita compensar desventaja)
            max_gimbal = 45.0 * np.pi / 180.0  # 45 grados
        else:
            # Límite estándar para otros casos
            max_gimbal = 30.0 * np.pi / 180.0  # 30 grados
        
        return np.clip(phi, -max_gimbal, max_gimbal)
    
    def _limit_gimbal_step(self, i, desired_ang_rad, max_step_jump_deg=4.0):
        """
        Limita el cambio de gimbal por timestep a ±max_step_jump_deg, y luego
        aplica el clamp absoluto existente (±20°) vía _clip_gimbal.
        Sugerencia realista: max_step_jump_deg = 0.5–1.0 (≈10–20°/s con dt=0.05s).
        """
        prev = 0.0
        if hasattr(self, 'engine_gimbals') and i < len(self.engine_gimbals):
            prev = float(self.engine_gimbals[i])

        max_jump = np.deg2rad(max_step_jump_deg)
        delta = float(desired_ang_rad) - prev

        if   delta >  max_jump: limited = prev + max_jump
        elif delta < -max_jump: limited = prev - max_jump
        else:                   limited = float(desired_ang_rad)

        # Mantener el límite absoluto existente ±20°:
        return self._clip_gimbal(limited)

    def _quantize_throttle(self, t):
        """
        Regla por motor:
        t <= 0  -> 0.0 (OFF)
        0 < t < 0.4 -> 0.4 (mínimo técnico)
        0.4 <= t <= 1.0 -> tal cual recortado
        """
        if t <= 0.0:
            return 0.0
        return float(np.clip(max(t, 0.4), 0.4, 1.0))

    def _distribute_throttle_across_engines(self, throttle):
        """
        Dado un throttle GLOBAL [0..1], decide cuántos motores encender
        (respetando disponibilidad de episodio) y el % por motor (>=40%).
        Preferencia: centro -> +lateral -> +lateral.
        """
        n = self.num_engines
        T = self.engine_thrust_sl
        avail = [available and operational and not locked_out
                 for available, operational, locked_out in zip(
                     self.engine_available,
                     self.engine_operational,
                     self.engine_locked_out,
                 )]
        if throttle <= 0.0 or T <= 0.0 or not any(avail):
            return [False]*n, [0.0]*n

        order = [1,0,2] if n==3 else list(range(n))
        avail_idx = [i for i in order if i<n and avail[i]]
        n_avail = len(avail_idx)

        # r_des = fracción vs. capacidad con n_avail motores
        r_des = float(np.clip(throttle, 0.0, 1.0))
        best_set, best_th, best_err = [], 0.0, 1e9
        candidates = []
        if n_avail>=1: candidates.append(avail_idx[:1])
        if n_avail>=2: candidates.append(avail_idx[:2])
        if n_avail>=3: candidates.append(avail_idx[:3])

        for S in candidates:
            k = len(S)
            th = r_des * (n_avail / k)
            if th > 0.0 and th < 0.4:
                # comparar "todo OFF" vs encender con 0.4
                err_off = abs(0.0 - r_des)
                r_04 = (k*0.4)/n_avail
                err_04 = abs(r_04 - r_des)
                if err_04 < err_off:
                    th = 0.4
                    r = r_04
                else:
                    th = 0.0
                    r = 0.0
                    k = 0
            else:
                th = 0.0 if th<=0.0 else float(np.clip(th, 0.4, 1.0))
                r = (k*th)/n_avail if th>0.0 else 0.0
            err = abs(r - r_des)
            if err < best_err:
                best_err = err
                best_set = S[:k] if th>0.0 else []
                best_th = th

        engines_on = [False]*n
        engines_thrust = [0.0]*n
        for i in best_set:
            engines_on[i] = True
            engines_thrust[i] = best_th * T
        return engines_on, engines_thrust
    
    def get_random_action(self):
        return random.randint(0, self.action_dims - 1)

    # def create_random_state(self):

    #     # predefined locations
    #     x_range = self.world_x_max - self.world_x_min
    #     y_range = self.world_y_max - self.world_y_min
    #     xc = (self.world_x_max + self.world_x_min) / 2.0
    #     yc = (self.world_y_max + self.world_y_min) / 2.0

    #     if self.task == 'landing':
    #         x = random.uniform(xc - x_range / 4.0, xc + x_range / 4.0)
    #         y = yc + 0.4*y_range
    #         if x <= 0:
    #             theta = -85 / 180 * np.pi
    #         else:
    #             theta = 85 / 180 * np.pi
    #         vy = -90

    #     if self.task == 'hover':
    #         x = xc
    #         y = yc + 0.2 * y_range
    #         theta = random.uniform(-45, 45) / 180 * np.pi
    #         vy = -10

    #     state = {
    #         'x': x, 'y': y, 'vx': 0, 'vy': vy,
    #         'theta': theta, 'vtheta': 0,
    #         'phi': 0, 'f': 0,
    #         't': 0, 'a_': 0,
    #         'engine_gimbals': [0.0, 0.0, 0.0]  # ← AÑADIR ESTO
    #     }

    #     return state

    def _create_curriculum_state(self):
        """Muestrea un spawn simetrico desde la fase seleccionada."""
        phase = self.curriculum_phase
        side = random.choice((-1.0, 1.0))

        x = side * random.uniform(*phase['abs_x_m'])
        altitude_agl = random.uniform(*phase['altitude_agl_m'])
        y = self.H / 2.0 + altitude_agl
        if y > self.world_y_max - 50.0:
            y = self.world_y_max - 100.0

        vx_limits = phase['inward_vx_mps']
        if phase.get('inward_vx_linear_with_abs_x', False):
            x_limits = phase['abs_x_m']
            vx_magnitude = float(np.interp(
                abs(x),
                x_limits,
                vx_limits,
            ))
        elif 'inward_vx_scale_per_x' in phase:
            vx_magnitude = float(np.clip(
                phase['inward_vx_scale_per_x'] * abs(x),
                vx_limits[0],
                vx_limits[1],
            ))
        else:
            vx_magnitude = random.uniform(*vx_limits)
        vx = -side * vx_magnitude
        vy = random.uniform(*phase['vy_mps'])

        theta_deg = random.uniform(*phase['abs_theta_deg'])
        theta_deg += np.random.uniform(*phase['theta_noise_deg'])
        theta = side * np.deg2rad(theta_deg)
        vtheta_magnitude = random.uniform(*phase['abs_vtheta_deg_s'])
        # The curriculum represents a flip already progressing from the
        # belly-flop attitude towards upright (theta=0).  Its angular velocity
        # must therefore oppose the current attitude sign.  Using the same
        # sign made both mirrored spawns initially rotate towards +/-180 deg.
        vtheta = -side * np.deg2rad(vtheta_magnitude)

        assert x * vx <= 1e-6, "vx no apunta hacia el centro"
        assert x * theta >= -1e-6, "theta no inclina hacia el centro"
        assert theta * vtheta <= 1e-6, "vtheta no reduce el angulo del flip"

        return {
            'x': x, 'y': y,
            'vx': vx, 'vy': vy,
            'theta': theta, 'vtheta': vtheta,
            'phi': 0.0, 'f': 0.0, 't': 0.0, 'a_': 0.0,
            'engine_gimbals': [0.0] * self.num_engines,
        }

    def create_random_state(self):
        """
        Crea estado inicial aleatorio para el episodio.
        
        CONFIGURACIÓN CON NUEVOS LÍMITES DEL MUNDO:
        - world_x: [-400, 400] = 800m
        - world_y: [-50, 1000] = 1050m
        
        Starship spawn:
        - Altitud inicial: 900m sobre el suelo (más realista)
        - Velocidad vertical: -90 m/s (terminal velocity)
        - Ángulo: 70-85° (belly flop)
        - Posición horizontal: ±200m del target
        """
        
        # V1 and isolated V2 share the symmetric curriculum distributions.
        # The recovered legacy three-engine spawn remains untouched below.
        if self.task == 'landing' and self.engine_mode in {'single', 'three_v2'}:
            return self._create_curriculum_state()

        x_range = self.world_x_max - self.world_x_min  # 800m
        y_range = self.world_y_max - self.world_y_min  # 1050m
        xc = (self.world_x_max + self.world_x_min) / 2.0  # 0m
        
        # Ground level (CoM del cohete en el suelo)
        # La física de contacto usa y=H/2 como centro de masas en el suelo.
        # V1 emplea esa misma referencia; la versión recuperada se conserva.
        ground_level = (
            self.H / 2.0
            if self.engine_mode in {'single', 'three_v2'}
            else self.world_y_min + self.H / 2.0
        )

        if self.task == 'landing':
            # ═══════════════════════════════════════════════════════════
            # POSICIÓN HORIZONTAL: ±200m del target
            # ═══════════════════════════════════════════════════════════
            rango = random.choice([(-25, -20), (20, 25)])
            x = random.uniform(*rango)
            
            # ═══════════════════════════════════════════════════════════
            # ALTITUD: 900m sobre el suelo
            # ═══════════════════════════════════════════════════════════
            # Con más espacio vertical, podemos empezar más alto
            # 900m = ~9% de los 10km reales de Starship
            
            target_altitude = random.uniform(80, 150)  # metros sobre el suelo
            y = ground_level + target_altitude
            # y = -25 + 900 = 875m ✅
            
            # Verificación de seguridad
            if y > self.world_y_max - 50:
                y = self.world_y_max - 100  # Margen de seguridad
            
            # --- Velocidad vertical: terminal ~ -90 m/s
            vy = random.uniform(-25.0, -12.0)

           
            # --- Velocidad horizontal: SIEMPRE hacia el target (x=0)
            min_vx_mag, max_vx_mag = 3.0, 5.0
            vx_mag = np.clip(0.06 * abs(x), min_vx_mag, max_vx_mag)
            if x > 0:
                vx = -vx_mag     # estás a la derecha -> muévete a la izquierda
            elif x < 0:
                vx = +vx_mag     # estás a la izquierda -> muévete a la derecha
            else:
                vx = random.uniform(-2.0, 2.0)

            # --- Ángulo belly flop 75–85°, apuntando hacia el target
            # Convención: +θ = inclinar a la izquierda (CCW).
            theta_deg = random.uniform(3.0, 8.0)
            if x > 0:
                theta_sign = +1.0   # a la derecha -> inclina a la izquierda
            elif x < 0:
                theta_sign = -1.0   # a la izquierda -> inclina a la derecha
            else:
                theta_sign = np.sign(vx) if vx != 0 else 0.0  # centrado: libre
            theta = theta_sign * (theta_deg * np.pi / 180.0)

            # --- (Opcional) pequeña variación para romper simetrías
            theta += (np.random.uniform(-2.0, 2.0) * np.pi / 180.0) * np.sign(theta if theta!=0 else 1)

            # --- Aserciones de coherencia (puedes desactivarlas en producción)
            if x != 0:
                assert x * vx <= 1e-6, "vx no apunta hacia el centro"
                assert x * theta >= -1e-6, "theta no inclina hacia el centro con convención +θ izquierda"

        elif self.task == 'hover':
            x = xc
            y = ground_level + 300.0
            theta = random.uniform(-45.0, 45.0) / 180.0 * np.pi
            vy = -10.0
            vx = 0.0

        state = {
            'x': x, 'y': y,
            'vx': vx, 'vy': vy,
            'theta': theta, 'vtheta': 0.0,
            'phi': 0.0, 'f': 0.0, 't': 0.0, 'a_': 0.0,
            'engine_gimbals': [0.0] * self.num_engines,
        }
        return state

    def check_crash(self, state):
        """
        Verifica si el cohete ha crasheado.
        
        MEJORADO: Usa velocidad de impacto guardada para detectar
        crashes por velocidad excesiva correctamente.
        """
        
        if self.task == 'hover':
            x, y = state['x'], state['y']
            theta = state['theta']
            crash = False
            if y <= self.H / 2.0:
                crash = True
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            return crash
    
        elif self.task == 'landing':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = math.hypot(vx, vy)
    
            crash = False
            
            # Crash si está en estabilización pero se vuelca
            if self.touchdown_stable_steps > 0:
                if abs(theta) >= 10/180*np.pi:  # >10° es volcado crítico
                    crash = True
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"💥 CRASH: Volcado durante estabilización ({math.degrees(theta):.1f}°)")
        
            # Crash si sale del mundo
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"💥 CRASH: Salió del mundo (y={y:.1f}m)")

            # El entorno recuperado solo comprobaba el techo. Sin estos
            # límites, el agente podía escapar horizontalmente y prolongar el
            # episodio sin intentar tocar suelo.
            if x <= self.world_x_min or x >= self.world_x_max:
                crash = True
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"💥 CRASH: Salió horizontalmente del mundo (x={x:.1f}m)")
            
            # ═══════════════════════════════════════════════════════════
            # 🔥 MEJORADO: Crash por condiciones al tocar suelo
            # ═══════════════════════════════════════════════════════════
            if y <= self.H / 2.0:  # Tocó suelo
                
                # MÉTODO 1: Usar velocidad de impacto guardada (PREFERIDO)
                if hasattr(self, 'just_touched') and self.just_touched:
                    impact_v = getattr(self, 'impact_velocity', 0.0)
                    impact_vy = getattr(self, 'impact_vy', 0.0)
                    impact_theta = getattr(self, 'impact_theta', 0.0)
                    
                    # A. Crash por velocidad total excesiva
                    if impact_v >= 7.0:
                        crash = True
                        if hasattr(self, 'debug_mode') and self.debug_mode:
                            print(f"💥 CRASH: Velocidad de impacto total {impact_v:.2f} m/s >= 7.0 m/s")
                    
                    # B. Crash por velocidad VERTICAL excesiva (más crítico)
                    if impact_vy >= 6.0:
                        crash = True
                        if hasattr(self, 'debug_mode') and self.debug_mode:
                            print(f"💥 CRASH: Velocidad vertical de impacto {impact_vy:.2f} m/s >= 6.0 m/s")
                
                # MÉTODO 2: Fallback si no hay velocidad guardada
                # (Menos confiable porque la física ya frenó)
                elif v >= 7.0:
                    crash = True
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"💥 CRASH: Velocidad post-impacto {v:.2f} m/s >= 7.0 m/s")
                
            # C. Crash por posición muy lejana del target
                if abs(x) >= 15.0:
                    crash = True
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"💥 CRASH: Fuera de zona de aterrizaje (x={x:.1f}m, límite=±15m)")
                
                # D. Crash por ángulo muy extremo
                if abs(theta) >= 10/180*np.pi:  # Más estricto: 10° en vez de 7°
                    crash = True
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"💥 CRASH: Ángulo excesivo ({math.degrees(theta):.1f}° >= 10°)")
                
                # E. Crash por rotación muy rápida
                if abs(vtheta) >= 5/180*np.pi:  # 5°/s
                    crash = True
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"💥 CRASH: Velocidad angular excesiva ({math.degrees(vtheta):.1f}°/s >= 5°/s)")

            # Ángulo y giro moderados pueden disiparse sobre las patas, pero
            # no se permite quedar indefinidamente en una banda intermedia.
            # Si en 2 s no inicia la ventana de estabilidad, es un touchdown
            # fallido y no un timeout artificialmente más favorable.
            if (
                self.touchdown_contact
                and self.touchdown_contact_step is not None
                and not self.success_display_active
                and not self.already_landing
                and self.step_id - self.touchdown_contact_step
                >= self.max_touchdown_settling_steps
            ):
                crash = True
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print("💥 CRASH: No alcanzó estabilidad tras 2.0 s de asentamiento")
    
            # Crash si penetra muy por debajo del suelo
            if y < self.H / 2.0 - 5.0:
                crash = True
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"💥 CRASH: Penetró el suelo (y={y:.1f}m)")
            
            return crash


    def _apply_ground_physics(self, theta, vtheta, y, vy, dt):
        """
        Aplica física de suelo: auto-enderezamiento o vuelco.
        
        Esta función calcula el torque adicional cuando el cohete está en el suelo,
        simulando el efecto de la gravedad sobre un cuerpo rígido apoyado.
        
        Args:
            theta: Ángulo actual (rad)
            vtheta: Velocidad angular actual (rad/s)
            y: Posición vertical del CoM (m)
            vy: Velocidad vertical (m/s)
            dt: Delta de tiempo (s)
        
        Returns:
            tau_ground: Torque a añadir (N·m)
            crash: Boolean indicando si debe crashear por vuelco
        """
        
        ground_level = self.H / 2.0  # 25m para Starship
        
        # Solo aplicar si está en el suelo o muy cerca
        if y > ground_level + 0.5:  # Más de 50cm sobre el suelo
            return 0.0, False
        
        # ═══════════════════════════════════════════════════════════════
        # GEOMETRÍA: Calcular estabilidad
        # ═══════════════════════════════════════════════════════════════
        
        altura_CoM = self.H / 2.0  # Altura del CoM desde el suelo
        
        # Brazo de palanca horizontal (proyección del CoM)
        # Cuando el cohete se inclina, el CoM se proyecta horizontalmente
        brazo_horizontal = altura_CoM * np.sin(theta)
        
        # Determinar si es estable (CoM dentro de la base)
        es_estable = abs(brazo_horizontal) < self.base_radius
        
        # Calcular ángulo crítico (donde empieza a volcar)
        angulo_critico = np.arctan(self.base_radius / altura_CoM)
        
        # ═══════════════════════════════════════════════════════════════
        # FÍSICA: Calcular torques según estabilidad
        # ═══════════════════════════════════════════════════════════════
        
        crash = False
        
        if es_estable:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # CASO ESTABLE: Auto-enderezamiento
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # Torque restaurador (proporcional al ángulo)
            # El cos(theta) hace que el torque sea máximo cuando está inclinado
            # y cero cuando está vertical
            tau_restore = -self.k_restore * theta * altura_CoM * np.cos(theta)
            
            # Fricción rotacional adaptativa
            # Más fricción cuando está casi vertical (evita oscilaciones)
            if abs(theta) < np.radians(5.0):  # < 5°
                c_friction = self.c_friction_stable * 2.0  # Extra fricción
            else:
                c_friction = self.c_friction_stable
            
            tau_friction = -c_friction * vtheta
            
            tau_ground = tau_restore + tau_friction
            
            # Debug opcional
            if self.debug_ground_physics:
                print(f"  [STABLE] θ={np.degrees(theta):.1f}°, "
                    f"brazo={brazo_horizontal:.2f}m, "
                    f"τ_restore={tau_restore/1e6:.1f}MN·m")
        
        else:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # CASO INESTABLE: Vuelco
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # Calcular cuánto ha sobrepasado el ángulo crítico
            overshoot = abs(theta) - angulo_critico
            
            # Torque de vuelco (acelera exponencialmente)
            # El signo de theta determina dirección del vuelco
            # Multiplicar por (1 + vtheta) para acelerar si ya está rotando
            tau_topple = np.sign(theta) * self.k_topple * overshoot * (1.0 + abs(vtheta) * 2.0)
            
            # Fricción reducida (más fácil volcar)
            tau_friction = -self.c_friction_topple * vtheta
            
            tau_ground = tau_topple + tau_friction
            
            # Crash si ha volcado demasiado
            if abs(theta) > np.radians(30.0):  # > 30°
                crash = True
                if self.debug_ground_physics:
                    print(f"  [CRASH] Volcado a {np.degrees(theta):.1f}°")
            
            # Debug opcional
            if self.debug_ground_physics and not crash:
                print(f"  [TOPPLING] θ={np.degrees(theta):.1f}°, "
                    f"brazo={brazo_horizontal:.2f}m > {self.base_radius:.2f}m, "
                    f"τ_topple={tau_topple/1e6:.1f}MN·m")
        
        # ═══════════════════════════════════════════════════════════════
        # CASO ESPECIAL: Alta velocidad angular cuando casi vertical
        # ═══════════════════════════════════════════════════════════════
        
        # Si está vertical pero girando muy rápido, puede volcar
        if abs(theta) < np.radians(5.0) and abs(vtheta) > 0.2:  # <5° pero >11°/s
            # Aplicar fricción extra para intentar frenar
            tau_extra_friction = -1e7 * np.sign(vtheta) * (abs(vtheta) ** 1.5)
            tau_ground += tau_extra_friction
            
            if self.debug_ground_physics:
                print(f"  [HIGH_SPIN] vθ={np.degrees(vtheta):.1f}°/s, aplicando fricción extra")
        
        return tau_ground, crash

    def check_landing_success(self, state):
        if self.task == 'landing':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5
            
            # Criterios ESTRICTOS para success
            return (
                y <= 0 + self.H / 2.0 and       # Toca suelo
                v < 5.0 and                     # Velocidad < 10 m/s
                abs(x) < 15.0 and                # Zona efectiva de touchdown: ±15 m
                abs(theta) < 5/180*np.pi and    # Ángulo < 10°
                abs(vtheta) < 3/180*np.pi       # Velocidad angular < 10°/s
            )

    def check_stable_landing(self, state):
        """
        Verifica si el cohete ha aterrizado y permanece estable.
    
        LÓGICA NUEVA:
        1. Aterrizar CON 1 motor (fase de aproximación)
        2. APAGAR motor una vez en el suelo
        3. Contar 20 steps con motores OFF
        4. Landing success
    
        Condiciones para CONTAR stable_steps:
        1. CoM toca el suelo (y <= H/2)
        2. Velocidad total < 5 m/s
        3. Centrado (|x| < 10 m)
        4. Ángulo < 5°
        5. Velocidad angular < 3°/s
        6. ⚠️ CRÍTICO: Motores OFF (<1% empuje total)
        7. Mantiene estabilidad 1 s (20 steps)
        """
    
        if self.task != 'landing':
            return False
    
        # --- Seguridad: no marcar aterrizaje en los primeros pasos ---
        if self.step_id < 10:
            return False
    
        # Extraer estado
        x, y = state['x'], state['y']
        vx, vy = state['vx'], state['vy']
        theta = state['theta']
        vtheta = state['vtheta']
        v = math.hypot(vx, vy)
    
        # --- Nivel del suelo ---
        ground_y = self.H / 2.0  # el CoM toca suelo a y = 25m (para H=50)
    
        # --- Altura sobre el suelo ---
        altitude = y - ground_y
        on_ground = (altitude <= 0.2) and (altitude >= -0.1)  # 20 cm de margen
    
        # ═══════════════════════════════════════════════════════════════
        # 🔥 NUEVA LÓGICA: Solo contar si TODOS los motores están OFF
        # ═══════════════════════════════════════════════════════════════
        total_thrust = sum(self.engines_thrust)
        
        # Threshold estricto: <1% del empuje máximo total
        # Con 3 motores: 3 × 2MN × 0.01 = 60,000 N
        engines_all_off = total_thrust < 0.01 * self.num_engines * self.engine_thrust_sl
    
        # --- Condiciones de estabilidad ---
        # TODAS las condiciones físicas DEBEN cumplirse
        # PERO solo cuenta si engines_all_off = True
        is_stable = (
            on_ground and
            v < 5.0 and
            abs(x) < 15.0 and
            abs(theta) < math.radians(5.0) and
            abs(vtheta) < math.radians(3.0) and
            engines_all_off  # 🔥 CRÍTICO: Solo cuenta con motores OFF
        )
    
        # --- Contador de estabilidad ---
        if is_stable:
            # Si cumple TODAS las condiciones (incluyendo motores OFF)
            if self.touchdown_start_step is None:
                self.touchdown_start_step = self.step_id
                # Debug opcional
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"✅ Inicio de conteo de estabilidad en step {self.step_id}")
                    print(f"   Estado: x={x:.1f}m, θ={math.degrees(theta):.1f}°, v={v:.1f}m/s")
                    print(f"   Motores OFF: {engines_all_off}")
    
            self.touchdown_stable_steps = self.step_id - self.touchdown_start_step
    
            if self.touchdown_stable_steps >= self.required_stable_steps:
                # Mantener un segundo el frame verde 20/20 antes de terminar
                # el episodio y permitir que aparezca en el vídeo.
                if self.success_display_start_step is None:
                    self.success_display_start_step = self.step_id
                    self.success_display_active = True
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print("✅ Estabilidad confirmada; mostrando éxito durante 1 s")
                elif (
                    self.step_id - self.success_display_start_step
                    >= self.success_display_steps
                ):
                    if hasattr(self, 'debug_mode') and self.debug_mode:
                        print(f"🎉 LANDING SUCCESS alcanzado en step {self.step_id}")
                        print(f"   Stable steps: {self.touchdown_stable_steps}/{self.required_stable_steps}")
                    return True
        else:
            # Si NO cumple alguna condición, resetear contador
            # ESTO ES CLAVE: Si tiene motor encendido, el contador se resetea
            if self.touchdown_stable_steps > 0:
                # Debug opcional: informar por qué se resetea
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"⚠️ Contador reseteado en step {self.step_id}")
                    print(f"   on_ground: {on_ground}, v<5: {v<5.0}, centered: {abs(x)<15.0}")
                    print(f"   vertical: {abs(theta)<math.radians(5.0)}, engines_off: {engines_all_off}")
            
            self.touchdown_stable_steps = 0
            self.touchdown_start_step = None
            self.success_display_start_step = None
            self.success_display_active = False
    
        return False
    
    def calculate_reward(
            self,
            state,
            action=None,
            engines_thrust=None,
            step_id=None,
            max_steps=None,
            **kwargs
        ):
        """
        Calcula el reward total y sus componentes.
        """
        # Inicialización perezosa del sistema de recompensas
        if not hasattr(self, 'reward_system') or self.reward_system is None:
            try:
                from improved_rewards import ImprovedRewardSystem
                self.reward_system = ImprovedRewardSystem(
                    task=self.task,
                    world_bounds={
                        'x_min': self.world_x_min, 'x_max': self.world_x_max,
                        'y_min': self.world_y_min, 'y_max': self.world_y_max
                    }
                )
                # CRÍTICO: Inicializar propiedades físicas
                self.reward_system.mass = self.mass
                self.reward_system.g = self.g
                self.reward_system.num_engines = self.num_engines
                self.reward_system.rocket_height = self.H
                self.reward_system._unused_zero_thr_streak = [0] * self.num_engines
            except Exception as e:
                print(f"Error inicializando reward system: {e}")
                return 0.0, {}

        # Normalizar entradas
        _action = action
        _engines_thrust = engines_thrust if engines_thrust is not None else getattr(self, 'engines_thrust', [0.0]*self.num_engines)
        _step_id = step_id if step_id is not None else getattr(self, 'step_id', 0)
        _max_steps = max_steps if max_steps is not None else getattr(self, 'max_steps', 800)
        _engine_operational = kwargs.get(
            'engine_operational', getattr(self, 'engine_operational', [True] * self.num_engines)
        )
        _engine_failed = kwargs.get(
            'engine_failed', getattr(self, 'engine_failed', [False] * self.num_engines)
        )
        _engine_locked_out = kwargs.get(
            'engine_locked_out', getattr(self, 'engine_locked_out', [False] * self.num_engines)
        )

        # CRÍTICO: Enriquecer el state con TODA la información
        enriched_state = dict(state)
        enriched_state['engines_thrust'] = _engines_thrust
        enriched_state['engine_gimbals'] = list(self.engine_gimbals)
        enriched_state['engine_operational'] = list(_engine_operational)
        enriched_state['engine_failed'] = list(_engine_failed)
        enriched_state['engine_locked_out'] = list(_engine_locked_out)
        enriched_state['curriculum_engine_mask'] = list(
            getattr(self, 'curriculum_engine_mask', [True] * self.num_engines)
        )
        enriched_state['total_thrust'] = float(sum(_engines_thrust))  # CRÍTICO: Calcular aquí
        enriched_state['engine_available'] = getattr(self, 'engine_available', [True] * self.num_engines)
        enriched_state['engines_on'] = getattr(self, 'engines_on', [False] * self.num_engines)
        enriched_state['touchdown_contact'] = bool(self.touchdown_contact)
        enriched_state['just_touched'] = bool(
            getattr(self, 'just_touched', False)
        )
        enriched_state['impact_velocity'] = float(
            getattr(self, 'impact_velocity', 0.0)
        )
        enriched_state['impact_theta'] = float(
            getattr(self, 'impact_theta', abs(float(state['theta'])))
        )
        enriched_state['mass'] = float(self.mass)  # CRÍTICO
        enriched_state['g'] = float(self.g)  # CRÍTICO
        
        # Formatear action
        formatted_action = None
        if _action is not None and isinstance(_action, (list, tuple, np.ndarray)):
            action_array = np.asarray(_action).ravel()
            if len(action_array) >= 3:
                formatted_action = {'throttles': action_array[:3].tolist()}
            else:
                formatted_action = {'throttles': [0.0] * self.num_engines}
        elif _engines_thrust:
            formatted_action = {
                'throttles': [t/self.engine_thrust_sl if self.engine_thrust_sl > 0 else 0.0 
                            for t in _engines_thrust]
            }
        else:
            formatted_action = {'throttles': [0.0] * self.num_engines}

        # Llamada principal
        try:
            total_reward, components = self.reward_system.calculate_reward(
                state=enriched_state,
                action=formatted_action,
                engines_thrust=_engines_thrust,
                step_id=_step_id,
                max_steps=_max_steps
            )
        except Exception as e:
            print(f"Error en calculate_reward: {e}")
            import traceback
            traceback.print_exc()
            total_reward = 0.0
            components = {}

        if components is None:
            components = {}

        # Recompensas terminales
        try:
            terminal = self.reward_system.get_terminal_rewards()
        except Exception:
            terminal = {'successful_landing': 50000.0, 'crash': -5000.0, 'timeout': -2000.0}

        # NUEVO: Detectar si terminó muy pronto (exploit)
        is_early_exit = False
        if (_step_id < 150) and (getattr(self, 'already_crash', False) or 
                                getattr(self, 'already_landing', False)):
            is_early_exit = True

        # Detectar si se salió del mundo
        out_of_bounds = False
        if getattr(self, 'already_crash', False):
            y = self.state.get('y', 0)
            if y >= self.world_y_max - 10:
                out_of_bounds = True

        if (
            getattr(self, 'already_landing', False)
            or getattr(self, 'success_display_active', False)
        ):
            # El estado verde se mantiene varios frames para el vídeo. El bonus
            # de aterrizaje debe entregarse una sola vez, no una vez por frame.
            if not getattr(self, 'landing_reward_granted', False):
                if is_early_exit and self.engine_mode == 'three':
                    # Aterrizaje válido pero sospechosamente rápido
                    tr = terminal.get('successful_landing', 50000.0) * (_step_id / 200.0)
                else:
                    # V1 uses low-altitude curriculum spawns, so an early
                    # touchdown is expected and must receive the full bonus.
                    tr = terminal.get('successful_landing', 50000.0)
                total_reward += tr
                components['terminal_landing'] = tr
                self.landing_reward_granted = True
        elif out_of_bounds:
            tr = -2000.0  # NUCLEAR
            total_reward += tr
            components['terminal_out_of_bounds'] = tr
        # elif is_early_exit:  # NUEVO
        #     tr = terminal.get('early_exit', -2000.0)
        #     total_reward += tr
        #     components['terminal_early_exit'] = tr
        elif getattr(self, 'already_crash', False):
            tr = terminal.get('crash', -500.0)
            total_reward += tr
            components['terminal_crash'] = tr
        elif _step_id >= _max_steps:
            # El limite se alcanza exactamente una vez. La condicion anterior
            # (max_steps - 1) cobraba el timeout en los steps 749 y 750.
            tr = terminal.get('timeout', -2000.0)  # 🔥 AUMENTADO de -200 a -2000
            total_reward += tr
            components['terminal_timeout'] = tr

        return total_reward, components  # ← CRÍTICO: AÑADIR ESTE RETURN

    def step(self, action):
        x, y = self.state['x'], self.state['y']
        vx, vy = self.state['vx'], self.state['vy']
        theta, vtheta = self.state['theta'], self.state['vtheta']
        phi = self.state['phi']

        continuous = isinstance(action, (list, tuple, np.ndarray)) and len(action) > 1
        
        if continuous and self.rocket_type == 'starship':
            a = np.array(action, dtype=np.float32)
            
            if self.engine_mode == 'three_v2' and len(a) != 9:
                raise ValueError(
                    f"V2 expects 9 actions, received {len(a)}"
                )

            if len(a) == 9 and self.engine_mode == 'three_v2':
                # Independent commands: three throttles, three physical
                # gimbal angles, and three ON/OFF signals. Separating the gate
                # from throttle lets the policy request minimum thrust without
                # approaching the shutdown boundary. Once an ignited engine is
                # commanded OFF it is locked out for the rest of the episode;
                # an engine that has never ignited remains available while OFF.
                throttle_commands = np.clip(a[:3], 0.0, 1.0)
                # The policy action is normalized to [-1, 1]. Map it
                # linearly to the physical +/-30 degree gimbal range before
                # applying the existing 1 degree-per-step slew-rate limit.
                desired_gimbals = (
                    np.clip(a[3:6], -1.0, 1.0)
                    * np.deg2rad(30.0)
                )
                on_commands = np.clip(a[6:9], -1.0, 1.0)
                new_on, new_thrust, new_gimbal = [], [], []

                for i in range(self.num_engines):
                    if (
                        not self.engine_operational[i]
                        or self.engine_locked_out[i]
                        or not self.curriculum_engine_mask[i]
                    ):
                        new_on.append(False)
                        new_thrust.append(0.0)
                        new_gimbal.append(0.0)
                        continue

                    was_on = bool(self.engines_on[i])
                    switch_command = float(on_commands[i])
                    force_on_in_flight = bool(
                        self.curriculum_phase.get(
                            'force_initial_engines_on_in_flight', False
                        )
                    ) and not self.touchdown_contact
                    if force_on_in_flight:
                        # Optional phase protection; touchdown still cuts
                        # all engines. Normal phases leave this flag false.
                        is_on = True
                    elif switch_command >= self.engine_switch_on_threshold:
                        is_on = True
                    elif switch_command <= self.engine_switch_off_threshold:
                        is_on = False
                    else:
                        # Schmitt-style latch: ambiguous commands preserve the
                        # current state, protecting an irreversible shutdown
                        # from small SAC exploration noise around zero.
                        is_on = was_on
                    if was_on and not is_on and not force_on_in_flight:
                        self.engine_locked_out[i] = True

                    if self.engine_locked_out[i] or not is_on:
                        new_on.append(False)
                        new_thrust.append(0.0)
                        new_gimbal.append(0.0)
                        continue

                    just_ignited = is_on and not was_on
                    throttle_command = float(throttle_commands[i])
                    if just_ignited:
                        throttle = self.min_engine_throttle
                    else:
                        throttle = self.min_engine_throttle + (
                            1.0 - self.min_engine_throttle
                        ) * throttle_command
                    new_on.append(is_on)
                    new_thrust.append(throttle * self.engine_thrust_sl)
                    new_gimbal.append(self._limit_gimbal_step(
                        i, float(desired_gimbals[i]), max_step_jump_deg=1.0
                    ))

                self.engines_on = new_on
                self.engines_thrust = new_thrust
                self.engine_gimbals = new_gimbal

            elif len(a) == 6:
                tvec = a[:3]  # throttles (0..1)
                vvec = a[3:]  # gimbals (rad)

                # CORRECCIÓN: Lógica simplificada y correcta para PERM OFF
                EPS = 1e-6
                
                # prepararemos nuevas listas
                new_on, new_thrust, new_gimbal = [], [], []

                for i in range(self.num_engines):
                    # Fallo físico o bloqueo nominal: ambos impiden encender,
                    # pero solo el primero cuenta como fallo de salud.
                    if not self.engine_operational[i] or self.engine_locked_out[i]:
                        new_on.append(False)
                        new_thrust.append(0.0)
                        new_gimbal.append(0.0)
                        continue
                    
                    # El motor está VIVO - procesar comando
                    ti_cmd = float(tvec[i])  # Comando de throttle 0..1
                    
                    was_on = self.engines_on[i] if hasattr(self, 'engines_on') else False
                    
                    if ti_cmd <= EPS:  # Comando de apagado
                        if was_on:
                            self.engine_locked_out[i] = True
                            # Debug opcional
                            if hasattr(self, 'debug_mode') and self.debug_mode:
                                print(f"Motor {i} -> LOCKED OUT (apagado nominal)")
                        # En cualquier caso, apagar
                        new_on.append(False)
                        new_thrust.append(0.0)
                        new_gimbal.append(0.0)
                        continue
                    
                    # Comando positivo - cuantizar
                    ti_q = self._quantize_throttle(ti_cmd)
                    
                    if ti_q <= EPS:  # Si después de cuantizar queda en 0
                        if was_on:
                            self.engine_locked_out[i] = True
                        new_on.append(False)
                        new_thrust.append(0.0)
                        new_gimbal.append(0.0)
                    else:
                        # Empuje válido - encender motor
                        new_on.append(True)
                        new_thrust.append(ti_q * self.engine_thrust_sl)
                        # new_gimbal.append(self._clip_gimbal(float(vvec[i])))
                        new_gimbal.append(self._limit_gimbal_step(i, float(vvec[i]), max_step_jump_deg=1.0))

                # Aplicar nuevas listas
                self.engines_on     = new_on
                self.engines_thrust = new_thrust
                self.engine_gimbals = new_gimbal

            elif len(a) == 9:
                # Acciones extendidas con ON/OFF
                tvec = a[:3]   # potencia normalizada (0–1)
                # Convertir gimbals de [-1,1] a radianes (±20°)
                max_gimbal = 30.0 * np.pi / 180.0
                vvec = np.clip(a[3:6], -1.0, 1.0) * max_gimbal  # gimbals en radianes

                onvec = a[6:9] # ON/OFF [-1, 1]

                EPS = 1e-6
                new_on, new_thrust, new_gimbal = [], [], []

                for i in range(self.num_engines):
                    # Un motor fallado y uno bloqueado no pueden encenderse.
                    if not self.engine_operational[i] or self.engine_locked_out[i]:
                        new_on.append(False)
                        new_thrust.append(0.0)
                        new_gimbal.append(0.0)
                        continue

                    # Señal ON/OFF
                    on_signal = float(onvec[i])
                    is_on_command = on_signal > 0
                    
                    # Verificar estado anterior (inicializar si no existe)
                    if not hasattr(self, 'engines_on') or i >= len(self.engines_on):
                        was_on = False
                    else:
                        was_on = self.engines_on[i]

                    # Si estaba encendido y ahora recibe OFF, queda bloqueado por
                    # la secuencia nominal de aterrizaje; no ha fallado.
                    if not is_on_command and was_on:
                        self.engine_locked_out[i] = True
                        if hasattr(self, 'debug_mode') and self.debug_mode:
                            print(f"Motor {i} → LOCKED OUT (apagado voluntariamente)")
                    
                    if not self.engine_operational[i] or self.engine_locked_out[i] or not is_on_command:
                        new_on.append(False)
                        new_thrust.append(0.0)
                        new_gimbal.append(0.0)
                        continue

                    # Motor encendido → throttle ya viene mapeado [0.4, 1.0]
                    ti_cmd = np.clip(tvec[i], 0.4, 1.0)  # Asegurar rango físico
                    throttle = ti_cmd  # Ya está en [0.4, 1.0]

                    new_on.append(True)
                    new_thrust.append(throttle * self.engine_thrust_sl)
                    new_gimbal.append(self._limit_gimbal_step(i, float(vvec[i]), max_step_jump_deg=1.0))

                self.engines_on = new_on
                self.engines_thrust = new_thrust
                self.engine_gimbals = new_gimbal


            elif len(a) == 2:
                throttle = float(np.clip(a[0], 0.0, 1.0))
                # v_all = self._clip_gimbal(float(a[1]))
                # self.engines_on, self.engines_thrust = self._distribute_throttle_across_engines(throttle)
                # self.engine_gimbals = [v_all if on else 0.0 for on in self.engines_on]
                v_all = float(a[1])  # sin clamp aquí: lo hace el limitador por step + _clip_gimbal
                if self.engine_mode == 'single':
                    # V1: actuador virtual continuo. No se cuantiza al 40 % ni
                    # se bloquea al apagarlo: cero significa simplemente sin empuje.
                    # El TVC puede preposicionarse y conservar su gimbal sin
                    # combustión, independientemente del empuje instantáneo.
                    active = throttle > 1e-6 and self.engine_operational[0]
                    self.engines_on = [active]
                    self.engines_thrust = [throttle * self.engine_thrust_sl if active else 0.0]
                    self.engine_gimbals = [
                        self._limit_gimbal_step(0, v_all, max_step_jump_deg=1.0)
                    ]
                else:
                    self.engines_on, self.engines_thrust = self._distribute_throttle_across_engines(throttle)
                    self.engine_gimbals = [
                        self._limit_gimbal_step(i, v_all, max_step_jump_deg=1.0) if on else 0.0
                        for i, on in enumerate(self.engines_on)]

            else:
                self.engines_on = [False]*self.num_engines
                self.engines_thrust = [0.0]*self.num_engines
                self.engine_gimbals = [0.0]*self.num_engines
            
            # Limpieza de residuos numéricos muy pequeños
            self.engines_thrust = [t if t > 1e-3 else 0.0 for t in self.engines_thrust]
            # engines_on = True solo si hay thrust positivo (estado instantáneo)
            self.engines_on = [t > 0.0 for t in self.engines_thrust]
            if self.engine_mode == 'three_v2':
                self.engine_throttles = [
                    float(np.clip(t / self.engine_thrust_sl, 0.0, 1.0))
                    for t in self.engines_thrust
                ]
            if self.engine_mode == 'three':
                self.engine_gimbals = [
                    g if on else 0.0
                    for g, on in zip(self.engine_gimbals, self.engines_on)
                ]

            f_total = float(np.sum(self.engines_thrust))
            vphi = 0.0

        else:
            # Modo discreto (Falcon o compatibilidad)
            if isinstance(action, (int, np.integer)):
                _, vphi = self.action_table[action]
            else:
                vphi = 0.0

            self.engines_on = [False]*self.num_engines
            self.engines_thrust = [0.0]*self.num_engines
            self.engine_gimbals = [0.0]*self.num_engines
            f_total = 0.0

        # Tras el primer contacto, esta tarea sólo modela aterrizaje: los
        # motores se cortan e ignoran mandos posteriores. Evita que el agente
        # convierta un touchdown en un nuevo despegue mientras se valida la
        # estabilidad sobre las patas.
        if self.task == 'landing' and self.touchdown_contact:
            self.engines_on = [False] * self.num_engines
            self.engines_thrust = [0.0] * self.num_engines
            self.engine_throttles = [0.0] * self.num_engines
            self.engine_gimbals = [0.0] * self.num_engines
            f_total = 0.0

        # ══════════════════════════════════════════════════════════════
        # 2. GEOMETRÍA (definir UNA VEZ)
        # ══════════════════════════════════════════════════════════════
        if self.engine_mode == 'three_v2':
            H = self.H
            nozzle_bases = [
                (offset, -H / 2.0)
                for offset in self.engine_x_offsets
            ]
        elif self.num_engines == 3:
            H, W = self.H, self.H/2.6
            nozzle_bases = [(-0.06*W, -H/2.0), (0.0, -H/2.0), (0.06*W, -H/2.0)]
        else:
            H, W = self.H, self.H/10
            nozzle_bases = [(0.0, -H/2.0)]
        
        # ══════════════════════════════════════════════════════════════
        # 3. CALCULAR MÉTRICAS (una sola vez)
        # ══════════════════════════════════════════════════════════════
        altitude = y - self.H/2.0
        num_engines_on = sum(1 for t in self.engines_thrust if t > 1e3)
        
        # Identificar motor lateral activo
        self.lateral_active = False
        self.active_idx = None
        if self.num_engines == 3 and num_engines_on == 1:
            self.active_idx = next((i for i, on in enumerate(self.engines_on) if on), None)
            if self.active_idx is not None and self.active_idx != 1:
                self.lateral_active = True
        
        # ══════════════════════════════════════════════════════════════
        # 4. COMPENSACIÓN AUTOMÁTICA DE GIMBAL (BLOQUE ÚNICO)
        # ══════════════════════════════════════════════════════════════
        for i in range(self.num_engines):
            T_i = self.engines_thrust[i]
            if T_i <= 1e3:
                continue
            
            # 🔥 FIX: Compensación REACTIVA proporcional a error
            if (
                self.engine_mode == 'three'
                and self.lateral_active
                and i == self.active_idx
            ):
                nx_base, ny_base = nozzle_bases[i]
                
                # ════════════════════════════════════════════════════════
                # NUEVA LÓGICA: Control proporcional al error de ángulo
                # ════════════════════════════════════════════════════════
                
                # Ganancia proporcional (negativa porque queremos oposición)
                # Si θ > 0 (inclinado derecha), motor izq debe girar positivo (derecha)
                # Si θ < 0 (inclinado izq), motor izq debe girar negativo (izq)
                Kp = -8.0  # rad de gimbal por rad de inclinación
                
                # Ganancia derivativa para anticipar movimiento
                Kd = -2.0  # rad de gimbal por rad/s de velocidad angular
                
                # Ángulo de gimbal objetivo (PD controller)
                target_compensation = Kp * theta + Kd * vtheta
                
                # Ajustar ganancia según altitud (más agresivo cerca del suelo)
                if altitude < 30:
                    # Zona crítica: máxima autoridad
                    target_compensation *= 1.5
                    max_gimbal_limit = np.deg2rad(45.0)  # Permitir más rango
                elif altitude < 50:
                    target_compensation *= 1.3
                    max_gimbal_limit = np.deg2rad(40.0)
                elif altitude < 100:
                    target_compensation *= 1.2
                    max_gimbal_limit = np.deg2rad(35.0)
                else:
                    # Altitud segura: más conservador
                    target_compensation *= 1.0
                    max_gimbal_limit = np.deg2rad(30.0)
                
                # Límites de seguridad
                target_compensation = np.clip(target_compensation, 
                                            -max_gimbal_limit, max_gimbal_limit)
                
                # ════════════════════════════════════════════════════════
                # INTERPOLACIÓN RÁPIDA (cambios más agresivos)
                # ════════════════════════════════════════════════════════
                current_gimbal = self.engine_gimbals[i]
                
                # Velocidad de convergencia adaptativa (más rápida que antes)
                if altitude < 30:
                    max_change_per_step = np.deg2rad(20.0)  # 20°/step (era 10°)
                elif altitude < 50:
                    max_change_per_step = np.deg2rad(15.0)  # 15°/step (era 7°)
                elif altitude < 100:
                    max_change_per_step = np.deg2rad(12.0)  # 12°/step (era 5°)
                else:
                    max_change_per_step = np.deg2rad(8.0)   # 8°/step (era 3°)
                
                delta = target_compensation - current_gimbal
                
                if abs(delta) > max_change_per_step:
                    # Cambiar gradualmente
                    smooth_gimbal = current_gimbal + np.sign(delta) * max_change_per_step
                else:
                    # Aplicar cambio completo (cerca del target)
                    smooth_gimbal = target_compensation
                
                # Aplicar con límites finales
                self.engine_gimbals[i] = np.clip(smooth_gimbal, 
                                                -max_gimbal_limit, max_gimbal_limit)
                

        
        # ══════════════════════════════════════════════════════════════
        # 5. CÁLCULO DE FUERZAS Y TORQUES (SIN compensación adicional)
        # ══════════════════════════════════════════════════════════════
        fx, fy, tau = 0.0, 0.0, 0.0
        ct, st = np.cos(theta), np.sin(theta)
        
        for i in range(self.num_engines):
            T_i = self.engines_thrust[i]
            if T_i <= 0.0:
                continue
            
            phi_i = self.engine_gimbals[i]  # ← YA compensado
            
            # Componentes en marco del cuerpo
            ft_i = -T_i * np.sin(phi_i)
            fr_i =  T_i * np.cos(phi_i)
            
            # Transformar a mundo
            Fx_i = ft_i*ct - fr_i*st
            Fy_i = ft_i*st + fr_i*ct
            
            # Torque
            nx, ny = nozzle_bases[i]
            rx = nx*ct - ny*st
            ry = nx*st + ny*ct
            tau_i = rx*Fy_i - ry*Fx_i
            
            fx += Fx_i
            fy += Fy_i
            tau += tau_i

        # =========================
        # AERODYNAMIC DRAG (CORREGIDO CON CoP)
        # =========================

        # Parámetros físicos realistas
        if not hasattr(self, 'rho_air'):
            self.rho_air = 1.225  # kg/m³ a nivel del mar

        if not hasattr(self, 'Cd_axial'):
            self.Cd_axial = 0.8  # Coeficiente de drag axial (forma cilíndrica)
            
        if not hasattr(self, 'Cd_lateral'):
            self.Cd_lateral = 1.4  # Coeficiente de drag lateral (cilindro perpendicular)

        if not hasattr(self, 'A_frontal'):
            # Área frontal: π·r² con diámetro proporcional a altura
            diameter = self.H / 5.5  # 50m / 5 = 10m (similar a Starship)
            self.A_frontal = np.pi * (diameter / 2.0) ** 2  # ≈78.5 m²

        if not hasattr(self, 'A_lateral'):
            # Área lateral proyectada
            diameter = self.H / 5.0
            self.A_lateral = self.H * diameter *0.9  # 50m × 10m = 500 m²

        if not hasattr(self, 'CoP_offset'):
            # Centro de Presión offset del CoM (metros)
            # Positivo = CoP por encima del CoM → efecto estabilizador
            self.CoP_offset = +0.15 * self.H  # ~4m (8% de la altura)

        if not hasattr(self, 'CoM_offset'):
            self.CoM_offset = -0.30 * self.H  # ~4m (8% de la altura)

        if not hasattr(self, 'c_omega'):
            # Damping rotacional realista
            self.c_omega = 3.0e6  # N·m·s (ajustado para I=29M kg·m²)

        if not hasattr(self, 'c_omega_q'):
            self.c_omega_q = 8.0e5  # N·m·s² (componente cuadrática)

        # ═══════════════════════════════════════════════════════════════
        # AERODINÁMICA: Drag + Torque estabilizador (modelo unificado)
        # ═══════════════════════════════════════════════════════════════

        rho = self.rho_air
        v_mag = np.hypot(vx, vy) + 1e-6  # magnitud de la velocidad
        flow_angle = np.arctan2(vy, vx)  # dirección del flujo

        if self.engine_mode in {'single', 'three_v2'}:
            # V1: proyección geométrica sin orientación izquierda/derecha.
            # El eje longitudinal del vehículo y el vector velocidad se
            # reflejan juntos al cambiar x, vx y theta de signo. Usar el
            # cuadrado de su producto vectorial garantiza el mismo drag en
            # ambos lados: 0 = vuelo axial, 1 = belly-flop transversal.
            body_axis_x = -np.sin(theta)
            body_axis_y = np.cos(theta)
            lateral_fraction = (
                body_axis_x * (vy / v_mag)
                - body_axis_y * (vx / v_mag)
            ) ** 2
            lateral_fraction = float(np.clip(lateral_fraction, 0.0, 1.0))
        else:
            # Modelo recuperado de tres motores. Se conserva sin cambios
            # para no invalidar su dinámica ni checkpoints legacy.
            target_aoa = np.deg2rad(80.0)
            angle_of_attack = (theta - flow_angle) - target_aoa
            angle_of_attack = (angle_of_attack + np.pi) % (2*np.pi) - np.pi
            lateral_fraction = np.sin(angle_of_attack) ** 2

        # Interpola suavemente entre drag axial y lateral.
        Cd_eff = self.Cd_axial + (self.Cd_lateral - self.Cd_axial) * lateral_fraction
        A_eff  = self.A_frontal + (self.A_lateral - self.A_frontal) * lateral_fraction

        # Fuerza total de drag (magnitud)
        F_drag = 0.5 * rho * Cd_eff * A_eff * v_mag**2

        # Dirección opuesta al flujo (en marco del mundo)
        Fx_drag = -F_drag * np.cos(flow_angle)
        Fy_drag = -F_drag * np.sin(flow_angle)

        # Aplicar drag total
        fx += Fx_drag
        fy += Fy_drag

        # ═══════════════════════════════════════════════════════════════
        # TORQUE AERODINÁMICO: efecto estabilizador (CoP > CoM)
        # ═══════════════════════════════════════════════════════════════

        lever_arm = (self.CoP_offset - self.CoM_offset)
        if self.engine_mode in {'single', 'three_v2'}:
            # Equilibrio en ±80°: el momento es restaurador y cero justo en
            # la referencia. De 450 m a 350 m se desvanece para liberar el
            # giro antes del landing burn simplificado.
            belly_ref = getattr(self, 'belly_flop_theta_reference', 0.0)
            belly_error = (theta - belly_ref + np.pi) % (2.0 * np.pi) - np.pi
            altitude = y - self.H / 2.0
            aero_hold = float(np.clip((altitude - 350.0) / 100.0, 0.0, 1.0))
            tau_aero = -lever_arm * F_drag * np.sin(2.0 * belly_error) * aero_hold
        else:
            # Compatibilidad con el modelo recuperado de tres motores.
            tau_aero = -lever_arm * F_drag * np.sin(angle_of_attack)

        # ---- Amortiguamiento angular ----
        tau_damp = (
            0.0
            if self.engine_mode == 'three_v2'
            else -self.angular_damping * vtheta
        )

        # Torque total (aerodinámico + amortiguamiento)
        tau += tau_aero + tau_damp

        # ═══════════════════════════════════════════════════════════════
        # DRAG ROTACIONAL: Amortiguamiento realista
        # ═══════════════════════════════════════════════════════════════

        altitude = y - self.H/2.0
        num_engines_on = sum(1 for t in self.engines_thrust if t > 1e3)

        # Identificar si hay motor lateral activo
        self.lateral_active = False
        if self.num_engines == 3 and num_engines_on == 1:
            self.active_idx = next((i for i, on in enumerate(self.engines_on) if on), None)
            if self.active_idx is not None and self.active_idx != 1:
                self.lateral_active = True

        if self.engine_mode == 'three_v2':
            # One smooth, mirror-symmetric damping model. It does not change
            # with altitude, active engine count, or selected engine side.
            tau_drag = (
                -self.c_omega * vtheta
                - self.c_omega_q * vtheta * abs(vtheta)
            )
            tau += tau_drag
        elif self.lateral_active:
            # 🔥 FIX: Damping reducido 6-7× para permitir control
            if altitude < 30:
                damping_multiplier = 3.0   # Reducido de 20.0 → ×6.7 menos
                quad_multiplier = 0.8      # Reducido de 3.0 → ×3.75 menos
            elif altitude < 50:
                damping_multiplier = 2.5   # Reducido de 15.0 → ×6 menos
                quad_multiplier = 0.7      # Reducido de 2.5 → ×3.6 menos
            elif altitude < 100:
                damping_multiplier = 2.0   # Reducido de 10.0 → ×5 menos
                quad_multiplier = 0.6      # Reducido de 2.0 → ×3.3 menos
            else:
                damping_multiplier = 1.5   # Reducido de 5.0 → ×3.3 menos
                quad_multiplier = 0.5      # Reducido de 1.5 → ×3 menos

        else:
            if altitude < 50 and num_engines_on == 2:
                damping_multiplier = 4.0
                quad_multiplier = 1.2
            elif altitude < 30:
                damping_multiplier = 5.0
                quad_multiplier = 1.5
            else:
                damping_multiplier = 1.0
                quad_multiplier = 1.0
        
        if self.engine_mode != 'three_v2':
            tau_drag = -(self.c_omega * damping_multiplier) * vtheta \
                    - (self.c_omega_q * quad_multiplier) * vtheta * abs(vtheta)
            tau += tau_drag

        # ═══════════════════════════════════════════════════════════════
        # 🔥 NUEVO: Física de suelo (auto-enderezamiento y vuelco)
        # ═══════════════════════════════════════════════════════════════
        
        # Solo aplicar si el cohete está en el suelo
        ground_level = self.H / 2.0
        if y <= ground_level + 0.5:  # En el suelo o muy cerca
            # Llamar a la función de física de suelo
            tau_ground, vuelco_crash = self._apply_ground_physics(
                theta=theta,
                vtheta=vtheta,
                y=y,
                vy=vy,
                dt=self.dt
            )
            
            # Añadir el torque de suelo al total
            tau += tau_ground
            
            # Si detectó vuelco, marcar como crash
            if vuelco_crash:
                self.already_crash = True
                
                if self.debug_ground_physics:
                    print(f"\n💥 CRASH POR VUELCO en step {self.step_id}")
                    print(f"   θ={np.degrees(theta):.1f}°, vθ={np.degrees(vtheta):.1f}°/s")

        # Dinámica translacional y rotacional
        ax = (fx) / self.mass
        ay = (fy) / self.mass - self.g
        atheta = tau / self.I
        
        atheta = tau / self.I

        if self.already_landing:
            vx = vy = ax = ay = theta = vtheta = atheta = 0.0
            phi = 0.0
            f_total = 0.0
            self.engines_on = [False]*self.num_engines
            self.engines_thrust = [0.0]*self.num_engines
            self.engine_throttles = [0.0]*self.num_engines
            self.engine_gimbals = [0.0]*self.num_engines

        # Integración (línea 1362-1367)
        self.step_id += 1
        x_new = x + vx*self.dt + 0.5*ax*(self.dt**2)
        y_new = y + vy*self.dt + 0.5*ay*(self.dt**2)
        vx_new, vy_new = vx + ax*self.dt, vy + ay*self.dt
        theta_new = theta + vtheta*self.dt + 0.5*atheta*(self.dt**2)
        vtheta_new = vtheta + atheta*self.dt

        # V1 uses a physical orientation, not an accumulated revolution
        # counter.  Keeping it in [-pi, pi] makes the trigonometric
        # observation and the terminal attitude checks describe the same
        # orientation.  Legacy remains untouched for checkpoint compatibility.
        if self.engine_mode in {'single', 'three_v2'}:
            theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
    
        # ══════════════════════════════════════════════════════════════
        # 🔥 NUEVO: Detección de impacto y guardado de velocidad
        # ══════════════════════════════════════════════════════════════
        ground_level = self.H / 2.0  # 25m para cohete de 50m
        
        # Detectar si el cohete va a tocar suelo en ESTE step
        # (estaba arriba del suelo y ahora está abajo/en el suelo)
        will_touch_ground = (y_new <= ground_level and y > ground_level)
        
        if will_touch_ground:
            self.touchdown_contact = True
            self.touchdown_contact_step = self.step_id
            # Guardar velocidad ANTES de que la física la modifique
            self.impact_velocity = math.hypot(vx_new, vy_new)
            self.impact_vy = abs(vy_new)
            self.impact_vx = abs(vx_new)
            self.impact_theta = abs(theta_new)
            self.just_touched = True
            
            # Debug opcional
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"\n⚠️ IMPACTO DETECTADO en step {self.step_id}")
                print(f"   Posición: x={x_new:.1f}m, y={y_new:.1f}m")
                print(f"   Velocidad total: {self.impact_velocity:.2f} m/s")
                print(f"   Velocidad vertical: {self.impact_vy:.2f} m/s")
                print(f"   Velocidad horizontal: {self.impact_vx:.2f} m/s")
                print(f"   Ángulo: {math.degrees(theta_new):.1f}°")
                
                # Predicción de crash
                will_crash = (
                    self.impact_velocity >= 7.0 or
                    self.impact_vy >= 6.0 or
                    abs(x_new) >= 15.0 or
                    abs(theta_new) >= 10/180*np.pi
                )
                if will_crash:
                    print(f"   💥 Predicción: CRASH")
                else:
                    print(f"   ✅ Predicción: OK")
        else:
            self.just_touched = False
        
        # ══════════════════════════════════════════════════════════════
        # LÍMITE FÍSICO DEL SUELO + REACCIÓN NORMAL (código existente)
        # ══════════════════════════════════════════════════════════════
        
        if y_new <= ground_level:
            # Cohete toca o atraviesa el suelo
            y_new = ground_level  # Clamp a nivel del suelo
            
            # Si venía cayendo, detener movimiento vertical
            if vy_new < 0:
                vy_new = 0.0  # Frenado instantáneo
            
            # Las patas absorben y disipan el movimiento residual. La fricción
            # anterior (0.95) dejaba al vehículo deslizar varios metros y
            # dificultaba completar el contador de estabilidad.
            vx_new *= 0.45
            vtheta_new *= 0.50
            
            # Si está prácticamente quieto, forzar a cero (evitar drift numérico)
            if abs(vx_new) < 0.05:
                vx_new = 0.0
            if abs(vy_new) < 0.01:
                vy_new = 0.0
            if abs(vtheta_new) < 0.001:
                vtheta_new = 0.0

        # Pequeño snap para evitar drift numérico en vtheta
        if abs(vtheta_new) < 1e-6 and abs(tau) < 1e-3:
            vtheta_new = 0.0

        # (phi solo para compat discreta)
        phi = phi + self.dt * vphi
        phi = max(phi, -30/180*np.pi)
        phi = min(phi,  30/180*np.pi)

        self.state = {
            'x': x_new, 'y': y_new, 'vx': vx_new, 'vy': vy_new,
            'theta': theta_new, 'vtheta': vtheta_new,
            'phi': phi, 'f': f_total,
            't': self.step_id, 'action_': action,
            'engine_gimbals': list(self.engine_gimbals)  # <<< NUEVO
        }

        self.state_buffer.append(self.state)

        # self.already_landing = self.check_landing_success(self.state)
        self.already_landing = self.check_stable_landing(self.state)
        self.already_crash = self.check_crash(self.state)
        reward, reward_components = self.calculate_reward(
            state=self.state,
            action=action,
            engines_thrust=self.engines_thrust,  # ← CRÍTICO: pasar los thrust actuales
            step_id=self.step_id,
            max_steps=self.max_steps,
            engine_operational=self.engine_operational,
            engine_failed=self.engine_failed,
            engine_locked_out=self.engine_locked_out,
            touchdown_stable_steps=self.touchdown_stable_steps  # 🔥 NUEVO
            )
        
        angle_deg = abs(self.state['theta']) * 180.0 / math.pi
        
        altitude = self.state['y'] - self.H/2.0
        
        if angle_deg > 120.0 and altitude > 50:  # Invertido y no cerca del suelo
            self.already_crash = True
            # Keep all V1 failure terminals on the reward system's scale. The
            # former direct -10000 spike destabilised the critic and bypassed
            # the normal terminal component accounting.
            try:
                terminal_crash = self.reward_system.get_terminal_rewards().get(
                    'crash', -500.0
                )
            except Exception:
                terminal_crash = -500.0
            if not any(
                key in reward_components
                for key in ('terminal_crash', 'terminal_out_of_bounds')
            ):
                reward += terminal_crash
                reward_components['terminal_crash'] = terminal_crash
            done = True
            info = {
                'termination_reason': 'inverted',
                'reward_components': reward_components,
            }
            return self.flatten(self.state), reward, done, info
        
        done = bool(self.already_crash or self.already_landing)
        info = {'reward_components': reward_components}
        return self.flatten(self.state), reward, done, info


    # def flatten(self, state):
    #     x = [state['x'], state['y'], state['vx'], state['vy'],
    #         state['theta'], state['vtheta'], state['t'],
    #         state['phi']]
    #     return np.array(x, dtype=np.float32)/100.  # Asegurar dtype

    def _v1_physical_observation(self, state):
        """Return the nine directly measurable state and actuator values."""
        altitude = max(0.0, float(state['y']) - self.H / 2.0)
        theta = float(state['theta'])
        current_throttle = float(np.clip(
            sum(getattr(self, 'engines_thrust', [0.0]))
            / max(self.engine_thrust_sl, 1.0),
            0.0,
            1.0,
        ))
        current_gimbal = float(getattr(self, 'engine_gimbals', [0.0])[0])

        return [
            np.clip(float(state['x']) / 200.0, -2.0, 2.0),
            np.clip(altitude / 600.0, 0.0, 2.0),
            np.clip(float(state['vx']) / 50.0, -2.0, 2.0),
            np.clip(float(state['vy']) / 100.0, -2.0, 2.0),
            np.sin(theta),
            np.cos(theta),
            np.clip(float(state['vtheta']), -2.0, 2.0),
            current_throttle,
            np.clip(current_gimbal / np.deg2rad(30.0), -1.0, 1.0),
        ]

    def _remaining_episode_fraction(self, state):
        return np.clip(
            1.0 - float(state['t']) / self.max_steps,
            0.0,
            1.0,
        )

    def _flatten_v1(self, state):
        """Return V1's 12 values, including two engineered guidance inputs."""
        altitude = max(0.0, float(state['y']) - self.H / 2.0)
        theta = float(state['theta'])

        # Same reference as the V1 reward. Its sign follows the initial
        # belly-flop side, preserving left/right mirror symmetry.
        flip_start = 450.0
        burn_start = 250.0
        if altitude >= flip_start:
            target_abs_theta = np.deg2rad(80.0)
        elif altitude > burn_start:
            blend = (altitude - burn_start) / (flip_start - burn_start)
            target_abs_theta = np.deg2rad(80.0) * blend
        else:
            target_abs_theta = 0.0

        belly_reference = getattr(self, 'belly_flop_theta_reference', theta)
        target_sign = -1.0 if belly_reference < 0.0 else 1.0
        target_theta = target_sign * target_abs_theta
        attitude_error = np.arctan2(
            np.sin(theta - target_theta),
            np.cos(theta - target_theta),
        )

        # Signed margin relative to the V1 braking safety envelope.
        # A negative value means that the braking manoeuvre is already late.
        downward_speed = max(0.0, -float(state['vy']))
        stopping_distance = max(
            0.0,
            (downward_speed ** 2 - 5.5 ** 2) / (2.0 * 20.0),
        )
        braking_margin = altitude - stopping_distance - 40.0
        observation = self._v1_physical_observation(state) + [
            np.clip(attitude_error / np.pi, -1.0, 1.0),
            np.clip(braking_margin / 100.0, -2.0, 2.0),
            self._remaining_episode_fraction(state),
        ]
        return np.asarray(observation, dtype=np.float32)

    def _flatten_v1_1(self, state):
        """Return V1.1's 10 values without engineered guidance inputs."""
        observation = self._v1_physical_observation(state) + [
            self._remaining_episode_fraction(state),
        ]
        return np.asarray(observation, dtype=np.float32)

    def _flatten_v2(self, state):
        """Compact V2: 14 physical features and three engine state codes.

        Engine codes: -1 unavailable (masked, failed or locked), 0 ready/OFF,
        +1 ON. Physical throttle is reported separately. VecNormalize learns
        the observation statistics; these fixed scales precede that wrapper.
        This reconstruction is incompatible with the split 23-input policy.
        """
        altitude = max(0.0, float(state['y']) - self.H / 2.0)
        x_norm = 2.0 * (float(state['x']) - self.world_x_min) / (
            self.world_x_max - self.world_x_min
        ) - 1.0
        y_norm = 2.0 * (float(state['y']) - self.world_y_min) / (
            self.world_y_max - self.world_y_min
        ) - 1.0
        gimbals = [
            np.clip(gimbal / np.deg2rad(30.0), -1.5, 1.5)
            for gimbal in self.engine_gimbals
        ]
        engine_states = [
            -1.0 if not enabled or not operational or locked_out
            else (1.0 if is_on else 0.0)
            for enabled, operational, locked_out, is_on in zip(
                self.curriculum_engine_mask,
                self.engine_operational,
                self.engine_locked_out,
                self.engines_on,
            )
        ]
        physical_throttles = [
            float(np.clip(value, 0.0, 1.0))
            for value in self.engine_throttles
        ]

        observation = [
            x_norm,
            y_norm,
            np.clip(float(state['vx']) / 50.0, -2.0, 2.0),
            np.clip(float(state['vy']) / 100.0, -2.0, 2.0),
            float(state['theta']) / np.pi,
            np.clip(float(state['vtheta']), -2.0, 2.0),
            np.clip(float(state['t']) / self.max_steps, 0.0, 1.0),
        ] + gimbals + [
            np.clip(altitude / 200.0, 0.0, 5.0),
        ] + physical_throttles + engine_states
        return np.asarray(observation, dtype=np.float32)

    def flatten(self, state):
        """
        Normaliza el estado para la red neuronal.

        MEJORAS v2.0:
        - Normalización basada en límites físicos del mundo
        - Información de configuración de motores
        - Flags de compensación y zona crítica
        """
        if self.observation_mode == 'v1_engineered':
            return self._flatten_v1(state)
        if self.observation_mode == 'v1_raw':
            return self._flatten_v1_1(state)
        if self.observation_mode == 'v2_raw':
            return self._flatten_v2(state)

        # Legacy three-engine observation starts here and remains unchanged.

        # ════════════════════════════════════════════════════════════
        # POSICIÓN (normalizada a límites del mundo)
        # ════════════════════════════════════════════════════════════
        x_range = self.world_x_max - self.world_x_min  # 800m
        y_range = self.world_y_max - self.world_y_min  # 1050m
        
        # Normalizar a [0, 1] y luego a [-1, 1]
        x_norm = 2 * (state['x'] - self.world_x_min) / x_range - 1
        y_norm = 2 * (state['y'] - self.world_y_min) / y_range - 1
        
        # ════════════════════════════════════════════════════════════
        # VELOCIDAD (con clipping de seguridad)
        # ════════════════════════════════════════════════════════════
        # Terminal velocity: ~90 m/s vertical, ~50 m/s horizontal
        vx_norm = np.clip(state['vx'] / 50.0, -2.0, 2.0)
        vy_norm = np.clip(state['vy'] / 100.0, -2.0, 2.0)
        
        # ════════════════════════════════════════════════════════════
        # ACTITUD (radianes)
        # ════════════════════════════════════════════════════════════
        theta_norm = state['theta'] / np.pi  # [-1, 1] para ±180°
        vtheta_norm = np.clip(state['vtheta'] / 1.0, -2.0, 2.0)  # rad/s
        
        # ════════════════════════════════════════════════════════════
        # TIEMPO Y PHI
        # ════════════════════════════════════════════════════════════
        t_norm = state['t'] / float(self.max_steps)
        phi_norm = state['phi'] / (30 * np.pi / 180)
        
        # ════════════════════════════════════════════════════════════
        # GIMBALS EFECTIVOS (actuales)
        # ════════════════════════════════════════════════════════════
        gimbals = state.get('engine_gimbals', [0.0, 0.0, 0.0])
        g_norm = [np.clip(g / (30*np.pi/180), -1.5, 1.5) for g in gimbals]
        
        # ════════════════════════════════════════════════════════════
        # 🔥 NUEVO: INFORMACIÓN DE CONTEXTO PARA COMPENSACIÓN
        # ════════════════════════════════════════════════════════════
        
        # Altitude normalizada (rango útil: 0-200m)
        altitude = state['y'] - self.H / 2.0
        altitude_norm = np.clip(altitude / 200.0, -0.5, 5.0)
        
        # Flag de zona crítica (30-100m: donde compensación es agresiva)
        in_critical_zone = 1.0 if 30 < altitude < 100 else 0.0
        
        # Estado instantáneo de combustión
        engines_on = getattr(self, 'engines_on', [False] * self.num_engines)
        engines_binary = [1.0 if on else 0.0 for on in engines_on]

        # Estados persistentes: salud física y bloqueo por secuencia.
        engine_operational = getattr(self, 'engine_operational', [True] * self.num_engines)
        engine_locked_out = getattr(self, 'engine_locked_out', [False] * self.num_engines)
        operational_binary = [1.0 if operational else 0.0 for operational in engine_operational]
        locked_binary = [1.0 if locked else 0.0 for locked in engine_locked_out]
        
        # Tipo de configuración de motores
        num_on = sum(engines_on)
        config_type = 0.0
        
        if num_on == 0:
            config_type = -1.0  # Peligro: todos OFF
        elif num_on == 1:
            self.active_idx = next((i for i, on in enumerate(engines_on) if on), None)
            if self.active_idx == 1:
                config_type = 0.33  # Motor central (simétrico, sin compensación)
            elif self.active_idx in [0, 2]:
                config_type = 0.67  # Motor lateral (compensación activa)
        else:
            config_type = 0.0  # Múltiples motores (normal)
        
        # ════════════════════════════════════════════════════════════
        # CONSTRUIR VECTOR DE OBSERVACIÓN
        # ════════════════════════════════════════════════════════════
        x = ([x_norm, y_norm, vx_norm, vy_norm,           # 4
            theta_norm, vtheta_norm, t_norm, phi_norm]  # 4
            + g_norm                                       # 3
            + [altitude_norm, in_critical_zone]            # 2
            + engines_binary                               # 3
            + operational_binary                            # 3
            + locked_binary                                 # 3
            + [config_type])                               # 1
        # Total: 23 dims
        
        return np.array(x, dtype=np.float32)

    def render(self, window_name='env', wait_time=1,
           with_trajectory=True, with_camera_tracking=True,
           crop_scale=0.4, return_rgb_array=False,
           render_layout='default'):

        if render_layout not in {'default', 'close_pad'}:
            raise ValueError(
                f"Unknown render layout {render_layout!r}; "
                "expected 'default' or 'close_pad'"
            )
        native_close_camera = render_layout == 'close_pad' and with_camera_tracking
        camera_bounds = (
            self.close_camera_bounds(crop_scale=0.20)
            if native_close_camera
            else None
        )
        canvas = (
            self.render_camera_background(camera_bounds)
            if native_close_camera
            else np.copy(self.bg_img)
        )
        polys = self.create_polygons()

        # draw target region
        for poly in polys['target_region']:
            self.draw_a_polygon(canvas, poly, camera_bounds=camera_bounds)
        # draw rocket
        for poly in polys['rocket']:
            self.draw_a_polygon(canvas, poly, camera_bounds=camera_bounds)
        frame_0 = canvas.copy()

        # draw engine work
        for poly in polys['engine_work']:
            self.draw_a_polygon(canvas, poly, camera_bounds=camera_bounds)
        frame_1 = canvas.copy()

        # The default layout preserves the recovered crop-and-upscale camera.
        # close_pad is already drawn directly at output resolution above.
        if with_camera_tracking and not native_close_camera:
            frame_0 = self.crop_alongwith_camera(
                frame_0, crop_scale=crop_scale
            )
            frame_1 = self.crop_alongwith_camera(
                frame_1, crop_scale=crop_scale
            )

        # draw trajectory
        if with_trajectory:
            self.draw_trajectory(frame_0)
            self.draw_trajectory(frame_1)

        if render_layout == 'close_pad':
            self.draw_pad_top_view(frame_0)
            self.draw_pad_top_view(frame_1)

        # draw text
        self.draw_text(frame_0, color=(0, 0, 0))
        self.draw_text(frame_1, color=(0, 0, 0))

        if return_rgb_array:
            bgr_frame = cv2.cvtColor(frame_1, cv2.COLOR_RGB2BGR)  # Primero RGB→BGR
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # Luego BGR→RGB
            return rgb_frame
        else:
            # Para visualización normal (OpenCV usa BGR)
            cv2.imshow(window_name, frame_0[:,:,::-1])  # CON [:,:,::-1] restaurado
            cv2.waitKey(wait_time)
            cv2.imshow(window_name, frame_1[:,:,::-1])  # CON [:,:,::-1] restaurado
            cv2.waitKey(wait_time)
            return frame_0, frame_1
        
    # FUNCIÓN AUXILIAR: Generar configuración de fallos de motores
    def _generate_engine_failure_config(self):
        """
        Genera configuración realista de fallos de motores para Starship.
        
        Reglas:
        - Solo 5% de episodios tienen fallos
        - Máximo 2 motores pueden fallar
        - Motor central más confiable (1% vs 3% laterales)
        - Nunca todos los motores fallan
        
        Returns:
            tuple: (engine_available_list, failed_engines_indices)
        """
        engine_available = [True] * self.num_engines  # Por defecto todos funcionan
        failed_engines = []

        if self.num_engines != 3:
            return engine_available, failed_engines
        
        # Solo 5% de probabilidad de tener fallos
        if np.random.rand() < 0:
            # Probabilidades individuales de fallo:
            # - Motores laterales (0=Izquierdo, 2=Derecho): 3% cada uno
            # - Motor central (1): 1% (más confiable)
            engine_fail_probs = [0.07, 0.04, 0.07]
            
            potential_failures = []
            for i, fail_prob in enumerate(engine_fail_probs):
                if np.random.rand() < fail_prob:
                    potential_failures.append(i)
            
            # Aplicar restricciones de seguridad
            if len(potential_failures) >= 3:
                # Si fallan los 3, mantener solo el motor central
                failed_engines = [0, 2]  # Fallan solo los laterales
            elif len(potential_failures) == 2:
                failed_engines = potential_failures
            elif len(potential_failures) == 1:
                failed_engines = potential_failures
            
            # Aplicar fallos
            for i in failed_engines:
                engine_available[i] = False
        
        return engine_available, failed_engines
        
    def create_polygons(self):

        polys = {'rocket': [], 'engine_work': [], 'target_region': []}

        if self.rocket_type == 'falcon':

            H, W = self.H, self.H/10
            dl = self.H / 30

            # rocket main body
            pts = [[-W/2, H/2], [W/2, H/2], [W/2, -H/2], [-W/2, -H/2]]
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})
            # rocket paint
            pts = utils.create_rectangle_poly(center=(0, -0.35*H), w=W, h=0.1*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            pts = utils.create_rectangle_poly(center=(0, -0.46*H), w=W, h=0.02*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            # rocket landing rack
            pts = [[-W/2, -H/2], [-W/2-H/10, -H/2-H/20], [-W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})
            pts = [[W/2, -H/2], [W/2+H/10, -H/2-H/20], [W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})

        elif self.rocket_type == 'starship':

            H, W = self.H, self.H / 2.6
            dl = self.H / 30

            # rocket main body (right half)
            pts = np.array([[ 0.        ,  0.5006878 ],
                           [ 0.03125   ,  0.49243465],
                           [ 0.0625    ,  0.48143053],
                           [ 0.11458334,  0.43878955],
                           [ 0.15277778,  0.3933975 ],
                           [ 0.2326389 ,  0.23796424],
                           [ 0.2326389 , -0.49931225],
                           [ 0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})

            # rocket main body (left half)
            pts = np.array([[-0.        ,  0.5006878 ],
                           [-0.03125   ,  0.49243465],
                           [-0.0625    ,  0.48143053],
                           [-0.11458334,  0.43878955],
                           [-0.15277778,  0.3933975 ],
                           [-0.2326389 ,  0.23796424],
                           [-0.2326389 , -0.49931225],
                           [-0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (212, 212, 232), 'edge_color': None})

            # upper wing (right)
            pts = np.array([[0.15972222, 0.3933975 ],
                           [0.3784722 , 0.303989  ],
                           [0.3784722 , 0.2352132 ],
                           [0.22916667, 0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # upper wing (left)
            pts = np.array([[-0.15972222,  0.3933975 ],
                           [-0.3784722 ,  0.303989  ],
                           [-0.3784722 ,  0.2352132 ],
                           [-0.22916667,  0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # lower wing (right)
            pts = np.array([[ 0.2326389 , -0.16368638],
                           [ 0.4548611 , -0.33562586],
                           [ 0.4548611 , -0.48555708],
                           [ 0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

            # lower wing (left)
            pts = np.array([[-0.2326389 , -0.16368638],
                           [-0.4548611 , -0.33562586],
                           [-0.4548611 , -0.48555708],
                           [-0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

        else:
            raise NotImplementedError('rocket type [%s] is not found, please choose one '
                                      'from (falcon, starship)' % self.rocket_type)

        # engine work
        f, phi = self.state['f'], self.state['phi']
        c, s = np.cos(phi), np.sin(phi)

        if self.rocket_type == 'starship':
            # Geometría Starship (H, W ya definidos antes en el bloque starship)
            # --- engine work (Starship) ---
            H, W = self.H, self.H/2.6
            dl = self.H / 30.0
            if self.engine_mode == 'three_v2':
                # Same offsets used by the V2 force/torque calculation.
                nozzle_bases = [
                    (offset, -H / 2.0)
                    for offset in self.engine_x_offsets
                ]
            elif self.num_engines == 1:
                nozzle_bases = [(0.0, -H / 2.0)]
            else:
                nozzle_bases = [
                    (-0.12 * W, -H / 2.0),
                    (0.0, -H / 2.0),
                    (0.12 * W, -H / 2.0),
                ]

            for i, (nx, ny) in enumerate(nozzle_bases):
                thrust_i = self.engines_thrust[i] if i < len(self.engines_thrust) else 0.0
                if thrust_i <= 0:
                    continue
                vphi_i = self.engine_gimbals[i] if i < len(self.engine_gimbals) else 0.0
                c_i, s_i = np.cos(vphi_i), np.sin(vphi_i)
                throttle_rel = np.clip(thrust_i / max(self.engine_thrust_sl, 1e-6), 0.0, 1.0)

                k1, k2, k3, k4 = 2, 5, 8, 12
                scale = 0.5 + 1.0 * throttle_rel
                sizes = [dl*(0.8+0.4*throttle_rel), 1.5*dl*(0.8+0.6*throttle_rel),
                        2.0*dl*(0.8+0.8*throttle_rel), 3.0*dl*(0.8+1.0*throttle_rel)]
                centers = [
                    (nx + (k1*dl*scale)*s_i, ny - (k1*dl*scale)*c_i),
                    (nx + (k2*dl*scale)*s_i, ny - (k2*dl*scale)*c_i),
                    (nx + (k3*dl*scale)*s_i, ny - (k3*dl*scale)*c_i),
                    (nx + (k4*dl*scale)*s_i, ny - (k4*dl*scale)*c_i),
                ]
                for (cx, cy), w in zip(centers, sizes):
                    pts = utils.create_rectangle_poly(center=(cx, cy), w=w, h=w)
                    polys['engine_work'].append({'pts': pts, 'face_color': (255,255,255), 'edge_color': None})

        else:

            if f > 0 and f < 0.5 * self.g:
                pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
                pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
                polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
            elif f > 0.5 * self.g and f < 1.5 * self.g:
                pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
                pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
                pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
                polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
            elif f > 1.5 * self.g:
                pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
                pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
                pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
                pts4 = utils.create_rectangle_poly(center=(12 * dl * s, -H / 2 - 12 * dl * c), w=3 * dl, h=3 * dl)
                polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
                polys['engine_work'].append({'pts': pts4, 'face_color': (255, 255, 255), 'edge_color': None})
            # target region
        if self.task == 'hover':
            pts1 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=0, h=self.target_r/3.0)
            pts2 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=self.target_r/3.0, h=0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
        else:
            pts1 = utils.create_ellipse_poly(center=(0, 0), rx=self.target_r, ry=self.target_r/4.0)
            pts2 = utils.create_rectangle_poly(center=(0, 0), w=self.target_r/3.0, h=0)
            pts3 = utils.create_rectangle_poly(center=(0, 0), w=0, h=self.target_r/6.0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts3, 'face_color': None, 'edge_color': (242, 242, 242)})

            # apply transformation
        for poly in polys['rocket'] + polys['engine_work']:
            M = utils.create_pose_matrix(tx=self.state['x'], ty=self.state['y'], rz=self.state['theta'])
            pts = np.array(poly['pts'])
            pts = np.concatenate([pts, np.ones_like(pts)], axis=-1)  # attach z=1, w=1
            pts = np.matmul(M, pts.T).T
            poly['pts'] = pts[:, 0:2]

        return polys
    
    def close_camera_bounds(self, crop_scale=0.20):
        """Return a rocket-following camera window in world coordinates.

        The window matches the field of view of the old crop-based camera,
        but is used as the input transform before rasterisation.  Clamping the
        centre reproduces the previous behaviour close to the world edges.
        """
        world_width = float(self.world_x_max - self.world_x_min)
        world_height = float(self.world_y_max - self.world_y_min)
        half_width = world_width * float(crop_scale)
        half_height = world_height * float(crop_scale)

        centre_x = float(np.clip(
            self.state['x'],
            self.world_x_min + half_width,
            self.world_x_max - half_width,
        ))
        centre_y = float(np.clip(
            self.state['y'],
            self.world_y_min + half_height,
            self.world_y_max - half_height,
        ))
        return (
            centre_x - half_width,
            centre_x + half_width,
            centre_y - half_height,
            centre_y + half_height,
        )

    def _load_close_background_source(self):
        """Load and cache the original background without first downsizing it."""
        if not self._close_bg_source_loaded:
            source = cv2.imread(str(self.path_to_bg_img), cv2.IMREAD_COLOR)
            self._close_bg_source = (
                cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
                if source is not None
                else None
            )
            self._close_bg_source_loaded = True
        return self._close_bg_source

    def render_camera_background(self, camera_bounds):
        """Render the close-camera background directly at output resolution."""
        x_min, x_max, y_min, y_max = camera_bounds
        source = self._load_close_background_source()
        if source is None:
            # Match utils.load_bg_img's fallback sky while evaluating its
            # vertical gradient over the camera's world-coordinate window.
            top_rgb = np.array([10, 24, 45], dtype=np.float32)
            bottom_rgb = np.array([150, 185, 215], dtype=np.float32)
            world_y = np.linspace(
                y_max, y_min, self.viewport_h, dtype=np.float32
            )
            blend = (
                (self.world_y_max - world_y)
                / float(self.world_y_max - self.world_y_min)
            )
            blend = np.clip(blend, 0.0, 1.0)[:, None]
            rows = top_rgb[None, :] * (1.0 - blend) + bottom_rgb[None, :] * blend
            return np.repeat(
                rows[:, None, :], self.viewport_w, axis=1
            ).astype(np.uint8)

        source_h, source_w = source.shape[:2]
        world_width = float(self.world_x_max - self.world_x_min)
        world_height = float(self.world_y_max - self.world_y_min)

        source_x1 = int(np.floor(
            (x_min - self.world_x_min) / world_width * source_w
        ))
        source_x2 = int(np.ceil(
            (x_max - self.world_x_min) / world_width * source_w
        ))
        # Image rows run downwards while world y runs upwards.
        source_y1 = int(np.floor(
            (self.world_y_max - y_max) / world_height * source_h
        ))
        source_y2 = int(np.ceil(
            (self.world_y_max - y_min) / world_height * source_h
        ))
        source_x1 = int(np.clip(source_x1, 0, source_w - 1))
        source_x2 = int(np.clip(source_x2, source_x1 + 1, source_w))
        source_y1 = int(np.clip(source_y1, 0, source_h - 1))
        source_y2 = int(np.clip(source_y2, source_y1 + 1, source_h))
        region = source[source_y1:source_y2, source_x1:source_x2]
        return cv2.resize(
            region,
            (self.viewport_w, self.viewport_h),
            interpolation=cv2.INTER_CUBIC,
        )

    def draw_a_polygon(self, canvas, poly, camera_bounds=None):

        pts, face_color, edge_color = poly['pts'], poly['face_color'], poly['edge_color']
        pts_px = self.wd2pxl(
            pts,
            camera_bounds=camera_bounds,
            viewport_shape=canvas.shape[:2],
        )
        if face_color is not None:
            cv2.fillPoly(canvas, [pts_px], color=face_color, lineType=cv2.LINE_AA)
        if edge_color is not None:
            cv2.polylines(canvas, [pts_px], isClosed=True, color=edge_color, thickness=1, lineType=cv2.LINE_AA)

        return canvas


    def wd2pxl(self, pts, to_int=True, camera_bounds=None,
               viewport_shape=None):

        pts = np.asarray(pts, dtype=np.float64)
        pts_px = np.zeros_like(pts, dtype=np.float64)

        if camera_bounds is None:
            x_min, x_max = self.world_x_min, self.world_x_max
            y_min, y_max = self.world_y_min, self.world_y_max
        else:
            x_min, x_max, y_min, y_max = camera_bounds

        if viewport_shape is None:
            viewport_h, viewport_w = self.viewport_h, self.viewport_w
        else:
            viewport_h, viewport_w = viewport_shape

        # Use one metres-to-pixels scale on both axes.  This preserves shape
        # and matches the original renderer's world projection exactly.
        scale_x = float(viewport_w) / float(x_max - x_min)
        scale_y = scale_x
        pts_px[:, 0] = (pts[:, 0] - x_min) * scale_x
        pts_px[:, 1] = viewport_h - (pts[:, 1] - y_min) * scale_y

        if to_int:
            return pts_px.astype(int)
        else:
            return pts_px

    def draw_text(self, canvas, color=(255, 255, 0)):
        def put_text(vis, text, pt):
            cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)

        pt = (10, 20)
        text = "simulation time: %.2fs" % (self.step_id * self.dt)
        put_text(canvas, text, pt)

        pt = (10, 40)
        text = "simulation steps: %d" % (self.step_id)
        put_text(canvas, text, pt)

        pt = (10, 60)
        # y es la coordenada del centro de masas; para el piloto y los vídeos
        # resulta más útil mostrar la altitud AGL, que vale 0 al tocar suelo.
        altitude_agl = self.state['y'] - self.H / 2.0
        text = "x: %.2f m, alt: %.2f m AGL" % (self.state['x'], altitude_agl)
        put_text(canvas, text, pt)

        pt = (10, 80)
        text = "vx: %.2f m/s, vy: %.2f m/s" % (self.state['vx'], self.state['vy'])
        put_text(canvas, text, pt)

        pt = (10, 100)
        text = "a: %.2f degree, va: %.2f degree/s" % \
            (self.state['theta'] * 180 / np.pi, self.state['vtheta'] * 180 / np.pi)
        put_text(canvas, text, pt)

        # ACTUALIZAR: Mostrar estado detallado de motores con nombres descriptivos
        if hasattr(self, 'engines_on') and hasattr(self, 'engines_thrust') and hasattr(self, 'engine_operational'):
            motor_names = ["Izq", "Ctr", "Der"] if self.num_engines == 3 else [f"M{i+1}" for i in range(self.num_engines)]
            
            for i, (on, thrust, operational, failed, locked_out) in enumerate(zip(
                self.engines_on, 
                self.engines_thrust, 
                self.engine_operational,
                self.engine_failed,
                self.engine_locked_out,
            )):
                pt = (10, 100 + 20*(i+1))
                motor_name = motor_names[i] if i < len(motor_names) else f"M{i+1}"
                
                # Determinar estado del motor
                if failed or not operational:
                    status = "FAILED"
                    color_status = (0, 0, 255)  # Rojo para fallido
                elif locked_out:
                    status = "LOCKED OUT"
                    color_status = (255, 165, 0)  # Naranja para apagado permanente
                elif on:
                    status = "ON"
                    color_status = (0, 255, 0)  # Verde para encendido
                else:
                    status = "OFF"
                    color_status = (255, 255, 0)  # Amarillo para apagado temporal
                
                thrust_pct = (thrust / max(self.engine_thrust_sl, 1e-6)) * 100.0
                
                # Mostrar con nombre descriptivo
                cv2.putText(canvas, f"Motor {motor_name}: {status}, empuje: {thrust_pct:5.1f}%", 
                        pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_status, 1, cv2.LINE_AA)

        # Mostrar estado de gimbals
        if hasattr(self, 'engine_gimbals'):
            motor_names = ["Izq", "Ctr", "Der"] if self.num_engines == 3 else [f"M{i+1}" for i in range(self.num_engines)]
            for i, ang in enumerate(self.engine_gimbals):
                pt = (10, 160 + 20*(i+1))
                motor_name = motor_names[i] if i < len(motor_names) else f"M{i+1}"
                put_text(canvas, f"Gimbal {motor_name}: {ang*180/np.pi:+5.1f}", pt)
        
        # NUEVO: Mostrar resumen de motores fallidos al inicio (mejorado)
        if hasattr(self, 'engines_failed_at_start') and self.engines_failed_at_start:
            pt = (10, 250)
            motor_labels = {0: "Izq", 1: "Ctr", 2: "Der"}
            failed_names = [motor_labels.get(i, f"M{i+1}") for i in self.engines_failed_at_start]
            operational = sum(getattr(self, 'engine_available', [True]*self.num_engines))
            
            # Color rojo para advertencia
            cv2.putText(canvas, f"MOTORES FALLIDOS: {', '.join(failed_names)} | Operativos: {operational}/3", 
                    pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
        # 🔥 NUEVO: Información de estabilización

        pt = (10, 280)  # Debajo de los gimbals
            
        # Mostrar estado de estabilización
        stable_steps = getattr(self, 'touchdown_stable_steps', 0)
        required_steps = getattr(self, 'required_stable_steps', 20)
        displayed_steps = min(stable_steps, required_steps)
            
        if stable_steps > 0:
            # En proceso de estabilización
            percentage = (displayed_steps / required_steps) * 100
            status_color = (0, 255, 255)  # Amarillo (en progreso)
                
            if stable_steps >= required_steps:
                status_text = f"STABLE: {displayed_steps}/{required_steps} COMPLETE"
                status_color = (0, 255, 0)  # Verde (éxito)
            else:
                status_text = f"STABILIZING: {displayed_steps}/{required_steps} ({percentage:.0f}%)"
                
            cv2.putText(canvas, status_text, pt, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, status_color, 2, cv2.LINE_AA)
                
            # Barra de progreso
            pt_bar = (10, 295)
            bar_width = 200
            bar_height = 10
                
            # Fondo de la barra (gris)
            cv2.rectangle(canvas, pt_bar, 
                        (pt_bar[0] + bar_width, pt_bar[1] + bar_height),
                        (100, 100, 100), -1)
                
            # Progreso (verde/amarillo)
            progress_width = int(bar_width * (displayed_steps / required_steps))
            if stable_steps >= required_steps:
                bar_color = (0, 255, 0)  # Verde
            else:
                bar_color = (0, 255, 255)  # Amarillo
                
            cv2.rectangle(canvas, pt_bar,
                        (pt_bar[0] + progress_width, pt_bar[1] + bar_height),
                        bar_color, -1)
                
            # Borde de la barra
            cv2.rectangle(canvas, pt_bar,
                        (pt_bar[0] + bar_width, pt_bar[1] + bar_height),
                        (255, 255, 255), 1)
            
        # 🔥 NUEVO: Indicador de SUCCESS
        if getattr(self, 'already_landing', False):
            pt_success = (10, 320)
            success_text = "✓ LANDING SUCCESS"
                
            # Texto verde brillante con fondo
            cv2.rectangle(canvas, 
                        (pt_success[0] - 5, pt_success[1] - 20),
                        (pt_success[0] + 200, pt_success[1] + 5),
                        (0, 100, 0), -1)  # Fondo verde oscuro
                
            cv2.putText(canvas, success_text, pt_success, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)  # Texto verde brillante

    def draw_trajectory(self, canvas, color=(255, 0, 0)):

        pannel_w, pannel_h = 256, 256
        traj_pannel = 255 * np.ones([pannel_h, pannel_w, 3], dtype=np.uint8)

        sw, sh = pannel_w/self.viewport_w, pannel_h/self.viewport_h  # scale factors

        # draw horizon line
        range_x, range_y = self.world_x_max - self.world_x_min, self.world_y_max - self.world_y_min
        pts = [[self.world_x_min + range_x/3, self.H/2], [self.world_x_max - range_x/3, self.H/2]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # draw vertical line
        pts = [[0, self.H/2], [0, self.H/2+range_y/20]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        if len(self.state_buffer) < 2:
            return

        # draw traj
        pts = []
        for state in self.state_buffer:
            pts.append([state['x'], state['y']])
        pts_px = self.wd2pxl(pts)

        dn = 5
        for i in range(0, len(pts_px)-dn, dn):

            x1, y1 = int(pts_px[i][0]*sw), int(pts_px[i][1]*sh)
            x1_, y1_ = int(pts_px[i+dn][0]*sw), int(pts_px[i+dn][1]*sh)

            cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x1_, y1_), color=color, thickness=2, lineType=cv2.LINE_AA)

        roi_x1, roi_x2 = self.viewport_w - 10 - pannel_w, self.viewport_w - 10
        roi_y1, roi_y2 = 10, 10 + pannel_h
        canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] = 0.6*canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] + 0.4*traj_pannel

    def draw_pad_top_view(self, canvas):
        """Draw a pad-relative top-view projection for the 2-D environment.

        The simulation has no lateral z coordinate, so the vehicle marker is
        projected onto the pad's horizontal diameter. This is a render-only
        aid and never enters observations, rewards, or dynamics.
        """
        canvas_h, canvas_w = canvas.shape[:2]
        panel_size = min(220, canvas_h - 20, canvas_w - 20)
        if panel_size < 120:
            return

        panel = np.full(
            (panel_size, panel_size, 3),
            (238, 242, 246),
            dtype=np.uint8,
        )
        center = (panel_size // 2, int(panel_size * 0.52))
        outer_radius = max(35, int(panel_size * 0.34))
        touchdown_radius = max(
            8,
            int(outer_radius * 15.0 / max(float(self.target_r), 1.0)),
        )

        cv2.putText(
            panel,
            "PAD TOP - 2D PROJECTION",
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(panel, center, outer_radius, (85, 85, 85), 2, cv2.LINE_AA)
        cv2.circle(
            panel, center, touchdown_radius, (35, 170, 65), 2, cv2.LINE_AA
        )
        cv2.line(
            panel,
            (center[0] - outer_radius, center[1]),
            (center[0] + outer_radius, center[1]),
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        cv2.line(
            panel,
            (center[0], center[1] - outer_radius),
            (center[0], center[1] + outer_radius),
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

        x = float(self.state['x'])
        vx = float(self.state['vx'])
        vy = float(self.state['vy'])
        theta = float(self.state['theta'])
        vtheta = float(self.state['vtheta'])
        display_range = max(float(self.target_r), 15.0)
        marker_x = int(round(
            center[0] + np.clip(x / display_range, -1.0, 1.0) * outer_radius
        ))
        marker = (marker_x, center[1])

        within_pad = abs(x) <= 15.0
        kinematics_safe = (
            np.hypot(vx, vy) < 5.0
            and abs(theta) < np.deg2rad(5.0)
            and abs(vtheta) < np.deg2rad(3.0)
        )
        if within_pad and kinematics_safe:
            marker_color = (35, 180, 70)
            status = "TOUCHDOWN ENVELOPE"
        elif within_pad:
            marker_color = (30, 170, 225)
            status = "PAD - UNSTABLE"
        else:
            marker_color = (215, 65, 65)
            status = "OUTSIDE +/-15 m"

        cv2.line(panel, center, marker, marker_color, 2, cv2.LINE_AA)
        cv2.circle(panel, marker, 6, marker_color, -1, cv2.LINE_AA)
        velocity_dx = int(round(np.clip(vx / 10.0, -1.0, 1.0) * 30.0))
        cv2.arrowedLine(
            panel,
            marker,
            (marker[0] + velocity_dx, marker[1]),
            (55, 95, 210),
            2,
            cv2.LINE_AA,
            tipLength=0.35,
        )
        cv2.putText(
            panel,
            f"x={x:+.1f} m  vx={vx:+.1f} m/s",
            (8, panel_size - 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            status,
            (8, panel_size - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            marker_color,
            1,
            cv2.LINE_AA,
        )

        x1 = canvas_w - panel_size - 10
        y1 = canvas_h - panel_size - 10
        roi = canvas[y1:y1 + panel_size, x1:x1 + panel_size]
        canvas[y1:y1 + panel_size, x1:x1 + panel_size] = (
            0.15 * roi + 0.85 * panel
        ).astype(np.uint8)



    def crop_alongwith_camera(self, vis, crop_scale=0.4):
        x, y = self.state['x'], self.state['y']
        xp, yp = self.wd2pxl([[x, y]])[0]
        crop_w_half, crop_h_half = int(self.viewport_w*crop_scale), int(self.viewport_h*crop_scale)
        # check boundary
        if xp <= crop_w_half + 1:
            xp = crop_w_half + 1
        if xp >= self.viewport_w - crop_w_half - 1:
            xp = self.viewport_w - crop_w_half - 1
        if yp <= crop_h_half + 1:
            yp = crop_h_half + 1
        if yp >= self.viewport_h - crop_h_half - 1:
            yp = self.viewport_h - crop_h_half - 1

        x1, x2, y1, y2 = xp-crop_w_half, xp+crop_w_half, yp-crop_h_half, yp+crop_h_half
        vis = vis[y1:y2, x1:x2, :]

        vis = cv2.resize(vis, (self.viewport_w, self.viewport_h))
        return vis
    
def debug_action(self, action):
    """Método de debugging para ver qué está haciendo cada motor"""
    print(f"Action received: {action}")
    print("Engine status:")
    for i in range(self.num_engines):
        status = "ON" if (i < len(self.engines_on) and self.engines_on[i]) else "OFF"
        thrust = self.engines_thrust[i] if i < len(self.engines_thrust) else 0.0
        gimbal = self.engine_gimbals[i] if i < len(self.engine_gimbals) else 0.0
        print(f"  Motor {i}: {status}, Thrust: {thrust/1e6:.2f}MN, Gimbal: {gimbal*180/np.pi:+.1f}°")
    print()
