"""Condiciones iniciales de los curricula de aterrizaje V1 y V2.

Las magnitudes usan unidades SI y las alturas son AGL (sobre el suelo).
El signo horizontal se elige de forma simetrica en cada episodio; ``vx`` y
``theta`` se orientan despues hacia el pad situado en x=0.
"""

from copy import deepcopy


CURRICULUM_PHASES = {
    "phase_1": {
        "description": "Touchdown basico: cerca del pad y casi vertical.",
        "altitude_agl_m": (80.0, 150.0),
        "abs_x_m": (20.0, 25.0),
        # La formula anterior clip(0.06 * abs(x), 3, 5) daba siempre 3 m/s
        # para |x| entre 20 y 25 m. Se conserva exactamente ese resultado.
        "inward_vx_mps": (3.0, 3.0),
        "vy_mps": (-25.0, -12.0),
        "abs_theta_deg": (3.0, 8.0),
        "theta_noise_deg": (-2.0, 2.0),
        "abs_vtheta_deg_s": (0.0, 0.0),
    },
    "phase_2": {
        "description": "Aproximacion intermedia: mas altura, offset y actitud.",
        "altitude_agl_m": (180.0, 300.0),
        "abs_x_m": (30.0, 50.0),
        "inward_vx_mps": (3.0, 5.0),
        "vy_mps": (-40.0, -25.0),
        "abs_theta_deg": (15.0, 30.0),
        "theta_noise_deg": (0.0, 0.0),
        # El signo se aplica en rocket.py en oposicion a theta para representar
        # un flip que ya progresa desde belly-flop hacia la vertical.
        "abs_vtheta_deg_s": (0.0, 4.0),
    },
    "phase_2_5": {
        "description": "Puente hacia belly-flop completo con dificultad solapada.",
        "altitude_agl_m": (300.0, 480.0),
        "abs_x_m": (45.0, 130.0),
        # Mantiene continuidad con phase 2 y se aproxima a los 9 m/s minimos
        # de phase 3 conforme aumenta el offset horizontal.
        "inward_vx_mps": (4.0, 9.0),
        "inward_vx_linear_with_abs_x": True,
        "vy_mps": (-70.0, -35.0),
        "abs_theta_deg": (25.0, 65.0),
        "theta_noise_deg": (0.0, 0.0),
        # rocket.py aplica el signo opuesto a theta para continuar el flip
        # hacia la vertical desde cualquiera de los dos lados.
        "abs_vtheta_deg_s": (0.0, 5.0),
    },
    "phase_3": {
        "description": "Escenario V1 completo anterior al curriculum.",
        "altitude_agl_m": (500.0, 600.0),
        "abs_x_m": (150.0, 200.0),
        # Mapea linealmente |x|=150..200 m a |vx|=9..15 m/s.
        "inward_vx_mps": (9.0, 15.0),
        "inward_vx_linear_with_abs_x": True,
        "vy_mps": (-95.0, -85.0),
        "abs_theta_deg": (75.0, 85.0),
        "theta_noise_deg": (-2.0, 2.0),
        "abs_vtheta_deg_s": (0.0, 0.0),
    },
    "v2_phase_1a": {
        "description": (
            "Aprendizaje terminal V2 con unicamente el motor central, "
            "inicialmente apagado y disponible, con apagado definitivo normal."
        ),
        "altitude_agl_m": (80.0, 120.0),
        "abs_x_m": (0.0, 8.0),
        "inward_vx_mps": (0.0, 1.0),
        "vy_mps": (-20.0, -12.0),
        "abs_theta_deg": (0.5, 3.0),
        "theta_noise_deg": (-0.5, 0.5),
        "abs_vtheta_deg_s": (0.0, 0.0),
        # izquierda, centro, derecha
        "engine_available_mask": (False, True, False),
        # The agent controls ignition and shutdown using the normal V2 latch.
        # OFF before ignition preserves availability; OFF after ignition locks.
        "initial_engine_on_mask": (False, False, False),
        "force_initial_engines_on_in_flight": False,
    },
    "v2_phase_1b": {
        "description": (
            "Adaptacion a tres motores con el mismo spawn de v2_phase_1a: "
            "todos inicialmente apagados y disponibles, con apagado definitivo normal."
        ),
        # Solo cambia la disponibilidad de motores respecto a v2_phase_1a.
        "altitude_agl_m": (80.0, 120.0),
        "abs_x_m": (0.0, 8.0),
        "inward_vx_mps": (0.0, 1.0),
        "vy_mps": (-20.0, -12.0),
        "abs_theta_deg": (0.5, 3.0),
        "theta_noise_deg": (-0.5, 0.5),
        "abs_vtheta_deg_s": (0.0, 0.0),
        "engine_available_mask": (True, True, True),
        "initial_engine_on_mask": (False, False, False),
        "force_initial_engines_on_in_flight": False,
    },
    "v2_phase_1c": {
        "description": (
            "Secuenciacion terminal con los tres motores inicialmente "
            "apagados y disponibles, con apagado irreversible."
        ),
        "altitude_agl_m": (160.0, 220.0),
        "abs_x_m": (10.0, 20.0),
        "inward_vx_mps": (1.0, 3.0),
        "vy_mps": (-65.0, -45.0),
        "abs_theta_deg": (2.0, 6.0),
        "theta_noise_deg": (-1.0, 1.0),
        "abs_vtheta_deg_s": (0.0, 0.0),
        "engine_available_mask": (True, True, True),
        "initial_engine_on_mask": (False, False, False),
        "force_initial_engines_on_in_flight": False,
    },
    "v2_phase_2": {
        "description": (
            "Flip intermedio con tres motores: atraviesa la desaparicion "
            "del soporte aerodinamico y enlaza con el burn terminal."
        ),
        # El soporte aerodinamico se desvanece entre 450 y 350 m AGL.
        "altitude_agl_m": (300.0, 460.0),
        "abs_x_m": (50.0, 120.0),
        "inward_vx_mps": (3.0, 8.0),
        "inward_vx_linear_with_abs_x": True,
        "vy_mps": (-70.0, -40.0),
        "abs_theta_deg": (20.0, 60.0),
        "theta_noise_deg": (-2.0, 2.0),
        # El signo se invierte en rocket.py para que el giro reduzca |theta|.
        "abs_vtheta_deg_s": (0.0, 5.0),
    },
    "v2_phase_3": {
        "description": (
            "Escenario V2 completo: belly flop alto, gran offset y "
            "velocidad cercana a la terminal."
        ),
        # Se solapa con el extremo dificil de phase 2 para facilitar el resume.
        "altitude_agl_m": (500.0, 600.0),
        "abs_x_m": (150.0, 200.0),
        # Mapea linealmente |x|=150..200 m a |vx|=9..15 m/s.
        "inward_vx_mps": (9.0, 15.0),
        "inward_vx_linear_with_abs_x": True,
        "vy_mps": (-95.0, -85.0),
        "abs_theta_deg": (75.0, 85.0),
        "theta_noise_deg": (-2.0, 2.0),
        "abs_vtheta_deg_s": (0.0, 0.0),
    },
}

# Nombre conservado para no romper comandos o experimentos anteriores. La
# nueva progresion explicita debe comenzar en v2_phase_1a.
CURRICULUM_PHASES["v2_phase_1"] = deepcopy(CURRICULUM_PHASES["v2_phase_1c"])
CURRICULUM_PHASES["v2_phase_1"]["description"] += " (alias compatible)"


# Cambiar solo esta constante selecciona la fase usada por defecto.
DEFAULT_CURRICULUM_PHASE = "phase_1"


def get_curriculum_phase(name):
    """Devuelve una copia de la fase y falla pronto si el nombre no existe."""
    try:
        return deepcopy(CURRICULUM_PHASES[name])
    except KeyError as exc:
        available = ", ".join(CURRICULUM_PHASES)
        raise ValueError(
            f"Fase de curriculum desconocida: {name!r}. Disponibles: {available}"
        ) from exc
