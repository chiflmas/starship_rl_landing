import numpy as np
import math
 
class ImprovedRewardSystem:
    """
    Sistema de rewards OPTIMIZADO para aterrizaje Starship.
    
    VERSIÓN CON FIXES APLICADOS:
    - Fix 3: Añadido engine_shutdown_bonus para incentivar apagado de motor
    - Todos los componentes originales mantenidos
    
    Filosofía:
    1. Recompensar control activo (motores encendidos)
    2. Recompensar descenso progresivo hacia el objetivo
    3. Recompensar estrategia de 1 motor (suicide burn)
    4. 🔥 NUEVO: Recompensar apagado correcto de motor en el suelo
    5. Penalizar solo comportamientos catastróficos
    """
    
    def __init__(self, task='landing', world_bounds=None):
        self.task = task
        self.world_bounds = world_bounds or {'x_min': -400, 'x_max': 400, 'y_min': -50, 'y_max': 1000}
        
        if task == 'landing':
            self.weights = {
                'alive_with_thrust': 1.0,
                'descending': 20.0,
                'attitude': 12.0,
                'approaching_target': 15.0,
                'velocity_control': 4.0,
                'throttle_efficiency': 3.0,
                'time_pressure': 15.0,
                'safety': 8.0,
                'dangerous_behavior': 10.0,
                'engine_health': 2.0,
                'engine_shutdown': 10.0,  # 🔥 FIX 3: NUEVO componente
            }
                    
            self.target_x, self.target_y = 0, 25
            self.rocket_height = 50.0
            self.mass = 140_000.0
            self.g = 9.81
            self.num_engines = 3
            self.engine_thrust_sl = 2e6  # Corregido: 2MN por motor (era 2.3e6)
            
            # Memoria para transiciones
            self._last_active_count = None
            
        elif task == 'hover':
            self.weights = {
                'distance': 3.0,
                'velocity': 1.5,
                'attitude': 1.5,
                'safety': 1.0,
            }
            self.target_x, self.target_y = 0, 200
            self.rocket_height = 50.0
            self.mass = 140_000.0
            self.g = 9.81
            self.num_engines = 3
 
    def calculate_reward(self, state, action=None, engines_thrust=None, 
                        step_id=None, max_steps=None, **kwargs):
        """
        Calcula reward total y componentes individuales.
        
        🔥 FIX 3: Añadido componente 'engine_shutdown'
        """
        reward_components = {}
        
        if self.task == 'landing':
            _engines_thrust = engines_thrust if engines_thrust is not None else state.get('engines_thrust', [0.0]*3)
            
            # Componentes principales
            reward_components['alive_with_thrust'] = self._alive_with_thrust_reward(state, _engines_thrust)
            reward_components['descending'] = self._descending_reward(state)
            reward_components['attitude'] = self._attitude_reward(state)
            reward_components['approaching_target'] = self._approaching_target_reward(state)
            reward_components['velocity_control'] = self._velocity_control_reward(state)
            reward_components['throttle_efficiency'] = self._throttle_efficiency_reward(state, _engines_thrust)
            reward_components['time_pressure'] = self._time_pressure_penalty(state, step_id, max_steps)
            reward_components['dangerous_behavior'] = self._dangerous_behavior_penalty(state, _engines_thrust)
            reward_components['safety'] = self._safety_penalty(state)
            reward_components['engine_health'] = self._engine_health_penalty(state)
            
            # 🔥 FIX 3: NUEVO - Bonus por apagar motor correctamente
            reward_components['engine_shutdown'] = self._engine_shutdown_bonus(state, _engines_thrust)
 
            
        else:  # hover
            reward_components['distance'] = self._distance_reward_hover(state)
            reward_components['velocity'] = self._velocity_reward_hover(state)
            reward_components['attitude'] = self._attitude_reward(state)
            reward_components['safety'] = self._safety_penalty(state)
        
        # Calcular total ponderado
        total_reward = sum(self.weights.get(key, 1.0) * value 
                          for key, value in reward_components.items())
        
        return total_reward, reward_components
    
    # ════════════════════════════════════════════════════════════════════════
    # 🔥 FIX 3: NUEVO COMPONENTE - Engine Shutdown Bonus
    # ════════════════════════════════════════════════════════════════════════
    
    def _engine_shutdown_bonus(self, state, engines_thrust):
        """
        Bonus por apagar motores correctamente una vez en el suelo.
        
        Lógica:
        - Si está en el suelo (<1m altura)
        - Y tiene motores OFF
        - Y está estable (velocidad baja, vertical)
        → Dar bonus significativo
        
        Esto incentiva al agente a:
        1. Aterrizar con 1 motor (control)
        2. Apagar motor inmediatamente (inicio de conteo)
        3. Mantener estabilidad (completar conteo)
        
        Returns:
            +15 a +25 puntos si motores OFF en suelo
            -2 puntos si motores ON innecesariamente en suelo
            0 puntos si no está en el suelo
        """
        altitude = state['y'] - 0.5 * self.rocket_height
        vy = state['vy']
        theta = state['theta']
        
        # Solo aplicar si está REALMENTE en el suelo
        if altitude > 1.0:
            return 0.0
        
        # Verificar que esté estable (sin contar motores)
        v = math.hypot(state['vx'], state['vy'])
        is_physically_stable = (
            v < 5.0 and
            abs(state['x']) < 10.0 and
            abs(theta) < math.radians(5.0) and
            abs(state.get('vtheta', 0)) < math.radians(3.0)
        )
        
        if not is_physically_stable:
            return 0.0
        
        # Verificar estado de motores
        thrusts = engines_thrust if isinstance(engines_thrust, (list, tuple)) else [engines_thrust]
        total_thrust = sum(thrusts)
        max_available = self.num_engines * self.engine_thrust_sl
        
        engines_all_off = total_thrust < 0.01 * max_available
        
        if engines_all_off:
            # 🎁 BONUS: Motores OFF en el suelo + estable
            # Este bonus se da POR STEP mientras mantiene esta configuración
            base_bonus = 15.0
            
            # Bonus adicional por estar MUY cerca del target
            if abs(state['x']) < 5.0:
                base_bonus += 5.0
            
            # Bonus adicional por estar MUY vertical
            if abs(theta) < math.radians(2.0):
                base_bonus += 5.0
            
            return base_bonus  # Hasta +25 por step
        else:
            # ⚠️ PENALIZACIÓN SUAVE: Motores encendidos innecesariamente en suelo
            # (no debe ser muy fuerte para no interferir con la aproximación)
            return -2.0
    
    # ════════════════════════════════════════════════════════════════════════
    # COMPONENTES ORIGINALES (sin cambios)
    # ════════════════════════════════════════════════════════════════════════
    
    def _engine_health_penalty(self, state):
        """
        Penaliza exclusivamente fallos físicos. Un motor apagado por la
        secuencia nominal puede estar bloqueado, pero sigue sano y no recibe
        penalización de salud.
        """
        engine_failed = state.get('engine_failed', [False, False, False])
        failed_now = int(sum(1 for failed in engine_failed if failed))
        prev = self._last_active_count if self._last_active_count is not None else failed_now
        new_failures = max(0, failed_now - prev)
 
        penalty_transition = -800.0 * new_failures
        step_cost = -4.0 * failed_now
 
        # Actualiza memoria para el siguiente step
        self._last_active_count = failed_now
        return penalty_transition + step_cost
    
    # ================================================================
    # COMPONENTE 1: ALIVE WITH THRUST
    # ================================================================
    
    def _alive_with_thrust_reward(self, state, engines_thrust):
        """
        Base por mantener control/impulso + preferencia estructural según altura.
        """
        altitude = state['y'] - 0.5 * self.rocket_height
        engine_operational = state.get('engine_operational', [True, True, True])
 
        thrusts = engines_thrust if isinstance(engines_thrust, (list, tuple)) else [engines_thrust]
        active_count = int(sum(1 for i, t in enumerate(thrusts)
                            if (i < len(engine_operational) and engine_operational[i]) and t > 0.10 * self.engine_thrust_sl))
 
        base = 0.0
 
        if active_count > 0:
            base = 1.0
            
            # Bonus suave por configuración óptima según altura
            altitude_factor = np.clip((altitude - 100) / 200, 0, 1)  # 0 a 100m, 1 a 300m+
            
            if active_count == 1:
                # 1 motor es mejor abajo
                base += 2.0 * (1 - altitude_factor)
            elif active_count == 2:
                # 2 motores OK en medio
                base += 1.0
            else:  # 3 motores
                # 3 motores mejor arriba
                base += 1.5 * altitude_factor
        
        return base
    
    # ================================================================
    # COMPONENTE 2: DESCENDING
    # ================================================================
    
    def _descending_reward(self, state):
        """
        VERSIÓN AGRESIVA: Forzar descenso continuo
        """
        altitude = state['y'] - 0.5 * self.rocket_height
        vy = state['vy']
        theta = state['theta']
        
        # Penalización MASIVA por frenar completamente
        if altitude > 100 and abs(vy) < 5.0:
            return -50.0 * (altitude / 100.0)  # Hasta -500 a 1000m
        
        # Penalización adicional si está invertido Y frenado
        angle_deg = abs(theta) * 180.0 / math.pi
        if angle_deg > 90.0 and vy > -10.0:
            return -100.0  # Doble penalización
        
        # Subiendo = catastrófico
        if vy > 0:
            penalty = -20.0 * (vy / 10.0)
            if altitude > 300:
                penalty *= 5.0
            return penalty
        
        # Objetivo CONTINUO sin saltos
        target_vy = -10 - 20 * (altitude / 500)  # -10 a 0m, -30 a 500m
        tolerance = 5 + 10 * (altitude / 500)     # ±5 a 0m, ±15 a 500m
        
        error = abs(vy - target_vy)
        if error < tolerance:
            return 5.0 * (1 - error/tolerance)
        else:
            return -2.0 * min(error/10, 5.0)
    
    # ================================================================
    # COMPONENTE 3: ATTITUDE
    # ================================================================
    
    def _attitude_reward(self, state):
        """
        VERSIÓN ANTI-INVERSIÓN: Penalización catastrófica por estar invertido
        """
        theta = state['theta']
        altitude = state['y'] - 0.5 * self.rocket_height
        
        # Normalizar [-π, π]
        while theta > math.pi:
            theta -= 2.0 * math.pi
        while theta < -math.pi:
            theta += 2.0 * math.pi
        
        angle_deg = abs(theta) * 180.0 / math.pi
        
        # CRÍTICO: Penalización NUCLEAR por inversión
        if angle_deg > 90.0:
            # Penalización exponencial por estar invertido
            overshoot = (angle_deg - 90.0) / 90.0  # 0 a 1
            base_penalty = -100.0 * (1.0 + overshoot * 2.0)  # -100 a -300
            
            # Escalar por altitud (peor si está alto)
            if altitude > 100:
                altitude_factor = 1.0 + (altitude / 200.0)
                base_penalty *= altitude_factor
            
            return base_penalty  # TERMINAR AQUÍ para inversiones
        
        # ═══════════════════════════════════════════════════════════
        # COMPORTAMIENTO NORMAL (no invertido)
        # ═══════════════════════════════════════════════════════════
        
        # Calcular base_reward basado en el ángulo
        if angle_deg < 5.0:
            # Vertical perfecto: reward máximo
            base_reward = 10.0
        elif angle_deg < 15.0:
            # Casi vertical: reward alto con decay cuadrático
            deviation = (angle_deg - 5.0) / 10.0  # 0→1
            base_reward = 10.0 - 4.0 * (deviation ** 2)
        elif angle_deg < 30.0:
            # Inclinación moderada: decay lineal
            deviation = (angle_deg - 15.0) / 15.0  # 0→1
            base_reward = 6.0 - 4.0 * deviation
        elif angle_deg < 45.0:
            # Inclinación significativa: reward pequeño
            deviation = (angle_deg - 30.0) / 15.0  # 0→1
            base_reward = 2.0 - 4.0 * deviation
        elif angle_deg < 70.0:
            # Muy inclinado: penalización moderada
            deviation = (angle_deg - 45.0) / 25.0  # 0→1
            base_reward = -2.0 - 6.0 * deviation
        else:  # 70° a 90°
            # Casi horizontal: penalización fuerte
            deviation = (angle_deg - 70.0) / 20.0  # 0→1
            base_reward = -8.0 - 7.0 * deviation
        
        # ═══════════════════════════════════════════════════════════
        # Scaling por altitud: MÁS CRÍTICO cerca del suelo
        # ═══════════════════════════════════════════════════════════
        
        if altitude < 50.0:
            # Últimos 50m: scaling x3
            altitude_factor = 3.0
        elif altitude < 100.0:
            # 50-100m: scaling x2
            altitude_factor = 2.0
        elif altitude < 200.0:
            # 100-200m: scaling x1.5
            altitude_factor = 1.5
        else:
            # >200m: sin scaling
            altitude_factor = 1.0
        
        final_reward = base_reward * altitude_factor
        
        return final_reward
    
    # ================================================================
    # COMPONENTE 4: APPROACHING TARGET
    # ================================================================
    
    def _approaching_target_reward(self, state):
        """
        Reward por acercarse al punto de aterrizaje.
        Penaliza estar lejos, alto sin descender, y hovering.
        """
        x = state['x']
        y = state['y']
        vy = state['vy']
        altitude = y - 0.5 * self.rocket_height
        
        dx = abs(x - self.target_x)
        
        # 1. Distancia horizontal
        if dx < 10:
            horizontal_reward = 5.0
        elif dx < 25:
            horizontal_reward = 3.0
        elif dx < 50:
            horizontal_reward = 1.0
        elif dx < 100:
            horizontal_reward = 0.0
        else:
            horizontal_reward = -2.0 * min(dx / 100.0, 3.0)
        
        # 2. Penalty por altura excesiva sin descender
        altitude_penalty = 0.0
        if altitude > 150:
            if vy >= -5:
                altitude_penalty = -5.0 * (altitude / 200.0)
                if dx > 50:
                    altitude_penalty *= 1.5
        
        # 3. Bonus por estar cerca del target Y bajo
        combined_bonus = 0.0
        if altitude < 100 and dx < 50:
            combined_bonus = 3.0 * (1.0 - dx/50.0) * (1.0 - altitude/100.0)
        
        return horizontal_reward + altitude_penalty + combined_bonus
    
    # ================================================================
    # COMPONENTE 5: VELOCITY CONTROL
    # ================================================================
    
    def _velocity_control_reward(self, state):
        """
        Reward por mantener velocidades controladas.
        """
        vx, vy = state['vx'], state['vy']
        altitude = state['y'] - 0.5 * self.rocket_height
        
        # Velocidad horizontal
        if abs(vx) < 5.0:
            vx_reward = 2.0
        elif abs(vx) < 15.0:
            vx_reward = 1.0
        else:
            vx_reward = -1.0
        
        # Velocidad vertical (depende de altura)
        if altitude < 50:
            # Cerca del suelo: debe frenar
            if -10 < vy < -2:
                vy_reward = 3.0
            elif -15 < vy < -10:
                vy_reward = 1.0
            else:
                vy_reward = -2.0
        elif altitude < 200:
            # Altura media: velocidad moderada
            if -30 < vy < -10:
                vy_reward = 2.0
            else:
                vy_reward = 0.0
        else:
            # Alto: puede caer rápido
            if -50 < vy < -20:
                vy_reward = 1.0
            else:
                vy_reward = 0.0
        
        return vx_reward + vy_reward
    
    # ================================================================
    # COMPONENTE 6: THROTTLE EFFICIENCY
    # ================================================================
    
    def _throttle_efficiency_reward(self, state, engines_thrust):
        """
        Reward por uso eficiente de throttle.
        Prefiere 1 motor al 40-70% cerca del suelo.
        """
        altitude = state['y'] - 0.5 * self.rocket_height
        
        thrusts = engines_thrust if isinstance(engines_thrust, (list, tuple)) else [engines_thrust]
        active_count = int(sum(1 for t in thrusts if t > 0.05 * self.engine_thrust_sl))
        
        if active_count == 0:
            return 0.0
        
        # Calcular ratio promedio de throttle
        active_thrusts = [t for t in thrusts if t > 0.05 * self.engine_thrust_sl]
        avg_ratio = sum(t / self.engine_thrust_sl for t in active_thrusts) / len(active_thrusts)
        
        r = 0.0
        
        # Preferencia según altura
        if altitude < 100:
            # Cerca del suelo: prefiere 1 motor al 40-70%
            if active_count == 1:
                if 0.50 <= avg_ratio <= 0.70:
                    r += 12.0
                elif avg_ratio < 0.40:
                    r -= 5.0
            else:
                r -= 8.0 * (active_count - 1)
        
        r = max(r, -20.0)
 
        return r
    
    # ================================================================
    # COMPONENTE 7: TIME PRESSURE
    # ================================================================
    
    def _time_pressure_penalty(self, state, step_id, max_steps):
        """
        VERSIÓN AGRESIVA: Fuerza descenso progresivo
        """
        if step_id is None or max_steps is None:
            return 0.0
        
        altitude = state['y'] - 0.5 * self.rocket_height
        vy = state['vy']
        
        progress = step_id / max_steps
        
        # NUEVO: Penalización exponencial por altitude excesiva
        if altitude > 200:
            altitude_penalty = -10.0 * (altitude / 500.0) * (progress ** 2)
        elif altitude > 100:
            altitude_penalty = -5.0 * (altitude / 200.0) * progress
        else:
            altitude_penalty = 0.0
        
        # NUEVO: Penalización SEVERA por hovering
        if abs(vy) < 3.0:  # Velocidad vertical muy baja
            hover_time_penalty = -20.0 * progress
            if altitude > 150:
                hover_time_penalty *= 2.0
        else:
            hover_time_penalty = 0.0
        
        # NUEVO: Bonus por progreso vertical
        expected_altitude = 500 * max(0, 1.0 - progress * 1.5)
        if altitude < expected_altitude:
            progress_bonus = 5.0 * (1.0 - altitude / max(expected_altitude, 1.0))
        else:
            progress_bonus = 0.0
        
        return altitude_penalty + hover_time_penalty + progress_bonus
    
    # ================================================================
    # COMPONENTE 8: DANGEROUS BEHAVIOR
    # ================================================================
    
    def _dangerous_behavior_penalty(self, state, engines_thrust):
        """
        Penaliza comportamientos peligrosos con mayor severidad cerca del suelo.
        - Exceso de throttle (umbral dinámico por altura)
        - Varios motores saturados cerca del suelo
        - Spin (vtheta) y tilt indirecto (mantenemos gimbal)
        - Velocidad vertical peligrosa baja
        """
        altitude = state['y'] - 0.5 * self.rocket_height
        vy = state['vy']
        vtheta = state['vtheta']
        engine_gimbals = state.get('engine_gimbals', [0.0, 0.0, 0.0])
 
        thrusts = engines_thrust if isinstance(engines_thrust, (list, tuple)) else [engines_thrust]
        total_thrust = float(sum(thrusts))
        n_eng = max(1, len(thrusts))
        max_total = self.engine_thrust_sl * n_eng
        thrust_ratio = 0.0 if max_total <= 0 else total_thrust / max_total
        active_count = int(sum(1 for t in thrusts if t > 0.05 * self.engine_thrust_sl))
 
        penalty = 0.0
 
        # 1) Exceso de throttle con umbral dinámico
        if altitude < 200.0:
            thr_excess_thr = 0.60
        elif altitude < 500.0:
            thr_excess_thr = 0.70
        else:
            thr_excess_thr = 0.80
 
        if thrust_ratio > thr_excess_thr:
            penalty -= 12.0 * ((thrust_ratio - thr_excess_thr) / max(1e-9, (1.0 - thr_excess_thr)))
 
        # 2) Varios motores ~a tope cerca del suelo
        if active_count >= 2 and altitude < 150.0:
            for t in thrusts:
                perc = 0.0 if self.engine_thrust_sl <= 0 else t / self.engine_thrust_sl
                if perc > 0.90:
                    penalty -= 8.0
                if perc > 0.98:
                    penalty -= 8.0  # extra si literalmente al tope
 
        # 4) Velocidad vertical peligrosa muy cerca del suelo
        if altitude < 100.0:
            if vy < -25.0:
                danger_factor = abs(vy + 25.0) / 15.0
                penalty -= 4.0 * danger_factor
            if vy < -40.0:
                penalty -= 10.0
 
        # 5) Spin descontrolado (más severo bajo)
        if altitude < 100.0:
            k_spin = 16.0
        elif altitude < 300.0:
            k_spin = 8.0
        else:
            k_spin = 3.0
        penalty -= k_spin * (vtheta ** 2)
 
        return penalty
    
    # ================================================================
    # COMPONENTE 9: SAFETY
    # ================================================================
    
    def _safety_penalty(self, state):
        """
        Penalty por acercarse a los límites del mundo.
        """
        x = state['x']
        y = state['y']
        
        penalty = 0.0
        margin = 100
        
        # Límites horizontales
        if x < self.world_bounds['x_min'] + margin:
            dist = (x - self.world_bounds['x_min']) / margin
            penalty -= (1 - dist) ** 2 * 5.0
        elif x > self.world_bounds['x_max'] - margin:
            dist = (self.world_bounds['x_max'] - x) / margin
            penalty -= (1 - dist) ** 2 * 5.0
        
        # Límite superior (techo)
        if y > self.world_bounds['y_max'] - margin:
            dist = (self.world_bounds['y_max'] - y) / margin
            penalty -= (1 - dist) ** 2 * 20.0
            if y > self.world_bounds['y_max'] - 20:
                penalty -= 100.0
        
        return penalty
    
    # ================================================================
    # HOVER MODE (simplificado)
    # ================================================================
    
    def _distance_reward_hover(self, state):
        dx = state['x'] - self.target_x
        dy = state['y'] - self.target_y
        distance = math.sqrt(dx*dx + dy*dy)
        return math.exp(-(distance**2) / (2 * 100**2)) - 0.2
    
    def _velocity_reward_hover(self, state):
        vx, vy = state['vx'], state['vy']
        v_total = math.sqrt(vx*vx + vy*vy)
        return math.exp(-v_total / 5.0) - 0.2
    
    # ================================================================
    # TERMINAL REWARDS
    # ================================================================
    
    def get_terminal_rewards(self):
        """
        Retorna recompensas terminales.
        
        NOTA: El timeout penalty se maneja en rocket.py (Fix 2: -2000)
        """
        if self.task == 'landing':
            return {
                'successful_landing': 50000.0,
                'crash': -500.0,  # Ajustado de -1000 a -500
                'timeout': -2000.0,  # 🔥 FIX 2: Aumentado de -500 a -2000
            }
        else:
            return {
                'successful_hover': 10000.0,
                'crash': -1000.0,
                'timeout': -500.0,
            }
    
    def reset_episode_memory(self):
        """Reset memoria entre episodios."""
        self._last_active_count = None
 
# Función de integración con rocket.py
def integrate_simplified_rewards(rocket_instance):
    """
    Integra el sistema de rewards simplificado con la instancia de Rocket.
    """
    rocket_instance.reward_system = ImprovedRewardSystem(
        task=rocket_instance.task,
        world_bounds={
            'x_min': rocket_instance.world_x_min,
            'x_max': rocket_instance.world_x_max,
            'y_min': rocket_instance.world_y_min,
            'y_max': rocket_instance.world_y_max
        }
    )
