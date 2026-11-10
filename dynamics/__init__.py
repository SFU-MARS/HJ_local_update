# Dynamics that the local updates can use, chosen by name with --dynamics.
# Each entry: the dimension, the full system class, and the subsystem classes of its decomposition.
#
# A full system class has
#   opt_ctrl_numpy(t, state, spat_deriv), dynamics_numpy(t, state, uOpt, dOpt)   for the numpy versions
#   opt_ctrl(t, state, spat_deriv), dynamics(t, state, uOpt, dOpt)               for the HeteroCL versions
# and, if a local update adds Lax-Friedrichs dissipation (3d),
#   dissipation_coeffs_numpy(t, state), dissipation_coeffs(t, state)
import inspect

from common.params import json_object
from dynamics.coupled_2d import couple_u
from dynamics.dubins_3d import dubins_car
from dynamics.quadrotor_6d import quadrotor
from subdynamics import coupled_2d as sub_coupled_2d
from subdynamics import dubins_3d as sub_dubins_3d
from subdynamics import quadrotor_6d as sub_quadrotor_6d

DYNAMICS = {
    "coupled_2d": {"dim": 2, "system": couple_u, "subsystems": (sub_coupled_2d.subsys,)},
    "dubins_3d": {"dim": 3, "system": dubins_car, "subsystems": (sub_dubins_3d.subsys_1, sub_dubins_3d.subsys_2)},
    "quadrotor_6d": {"dim": 6, "system": quadrotor, "subsystems": (sub_quadrotor_6d.subsys_1, sub_quadrotor_6d.subsys_2)},
}

# The control mode follows how the subsystems are combined:
#   union = avoid the set (control maximizes V), intersection = reach the set (control minimizes V)
U_MODE = {"union": "max", "intersection": "min"}


def add_dynamics_args(parser, default):
    '''
        Add --dynamics and --dynamics_params to an argparse parser of a local update script.
            default: default dynamics name of the script, e.g. "dubins_3d"
    '''
    parser.add_argument("--dynamics", choices=list(DYNAMICS), default=default,
                        help=f"Dynamics of the full system (default {default}). The control mode follows --combine: "
                        "union = avoid (uMode max), intersection = reach (uMode min).")
    parser.add_argument("--dynamics_params", type=json_object, default={},
                        help="Parameters of the dynamics as a JSON object, e.g. '{\"v\": 1, \"wMax\": 1}' for dubins_3d. "
                        "Anything left out takes the default of the dynamics class.")


def check_dynamics(name, dim):
    '''
        Raise an error if name is not in DYNAMICS or its dimension is not dim.
    '''
    if name not in DYNAMICS:
        raise ValueError(f"unknown dynamics {name!r}, choose from {list(DYNAMICS)}")
    if DYNAMICS[name]["dim"] != dim:
        raise ValueError(f"dynamics {name!r} is {DYNAMICS[name]['dim']}d, this local update is {dim}d")


def _make(cls, params, combine):
    if "uMode" in params:
        raise ValueError("uMode is set by --combine (union: max, intersection: min), leave it out of dynamics_params")
    accepted = [p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "uMode")]
    unknown = sorted(set(params) - set(accepted))
    if unknown:
        raise ValueError(f"unknown parameters {unknown} for {cls.__name__}, it accepts {accepted}")
    kwargs = {"dMode": "min", **params, "uMode": U_MODE[combine]}
    return cls(**kwargs)


def make_dynamics(name, params, combine, dim):
    '''
        Full system dynamics by name.
            name: key of DYNAMICS, e.g. "dubins_3d"
            params: dict of constructor arguments, e.g. {"v": 1, "wMax": 1}; anything left out takes
                    the class default. uMode comes from combine, dMode defaults to "min".
            combine: "union" or "intersection"
            dim: dimension of the local update, checked against the dynamics
    '''
    check_dynamics(name, dim)
    return _make(DYNAMICS[name]["system"], params or {}, combine)


def make_subdynamics(name, params, combine, dim):
    '''
        Subsystem dynamics of the full system name, same parameters as make_dynamics.
        Returns a tuple with one object per subsystem class.
    '''
    check_dynamics(name, dim)
    return tuple(_make(cls, params or {}, combine) for cls in DYNAMICS[name]["subsystems"])
