"""Condiciones iniciales del curriculum de aterrizaje V1.

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
}


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
