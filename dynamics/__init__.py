# Dynamics that the local updates can use, chosen by name with --dynamics.
# Each entry: the dimension, the full system class, and the subsystem classes of its decomposition.
#
# A full system class has
#   opt_ctrl, opt_dstb, dynamics                     HeteroCL methods, for HJSolver (compute/) and the HeteroCL local updates
#   opt_ctrl_numpy, dynamics_numpy                   for the numpy local updates
#   dissipation_coeffs_numpy, dissipation_coeffs     Lax-Friedrichs coefficients as in HJSolver, from the optimal
#                                                    control at the global min/max derivatives (deriv_min, deriv_max)
#   DISSIPATION_USES_DERIV_BOUNDS                    True if those coefficients depend on deriv_min/deriv_max,
#                                                    then the local update computes them (common/hj_update.py)
#   DERIV_BOUNDS_DIMS (optional)                     the dimensions whose bounds the coefficients use (default all);
#                                                    only those are computed, the others are 0
# Subsystem classes need the HeteroCL methods opt_ctrl, opt_dstb, dynamics for HJSolver.
import inspect

from common.params import json_object
from dynamics.coupled_2d import couple_u
from dynamics.dubins_3d import dubins_car
from dynamics.dubins_4d import dubins_car_4d
from dynamics.quadrotor_6d import quadrotor
from subdynamics import coupled_2d as sub_coupled_2d
from subdynamics import dubins_3d as sub_dubins_3d
from subdynamics import dubins_4d as sub_dubins_4d
from subdynamics import quadrotor_6d as sub_quadrotor_6d

DYNAMICS = {
    "coupled_2d": {"dim": 2, "system": couple_u, "subsystems": (sub_coupled_2d.subsys,)},
    "dubins_3d": {"dim": 3, "system": dubins_car, "subsystems": (sub_dubins_3d.subsys_1, sub_dubins_3d.subsys_2)},
    "dubins_4d": {"dim": 4, "system": dubins_car_4d, "subsystems": (sub_dubins_4d.subsys_1, sub_dubins_4d.subsys_2)},
    "quadrotor_6d": {"dim": 6, "system": quadrotor, "subsystems": (sub_quadrotor_6d.subsys_1, sub_quadrotor_6d.subsys_2)},
}

def add_dynamics_args(parser, default):
    '''
        Add --dynamics and --dynamics_params to an argparse parser of a local update script.
            default: default dynamics name of the script, e.g. "dubins_3d"
    '''
    parser.add_argument("--dynamics", choices=list(DYNAMICS), default=default,
                        help=f"Dynamics of the full system (default {default}).")
    parser.add_argument("--dynamics_params", type=json_object, default={},
                        help="Parameters of the dynamics as a JSON object, e.g. '{\"v\": 1, \"wMax\": 1, \"uMode\": \"min\"}' for "
                        "dubins_3d. uMode is the control mode, 'max' (avoid, default) or 'min' (reach); "
                        "anything else left out takes the default of the dynamics class.")


def check_dynamics(name, dim):
    '''
        Raise an error if name is not in DYNAMICS or its dimension is not dim.
    '''
    if name not in DYNAMICS:
        raise ValueError(f"unknown dynamics {name!r}, choose from {list(DYNAMICS)}")
    if DYNAMICS[name]["dim"] != dim:
        raise ValueError(f"dynamics {name!r} is {DYNAMICS[name]['dim']}d, this local update is {dim}d")


def _make(cls, params):
    accepted = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
    unknown = sorted(set(params) - set(accepted))
    if unknown:
        raise ValueError(f"unknown parameters {unknown} for {cls.__name__}, it accepts {accepted}")
    kwargs = {"uMode": "max", "dMode": "min", **params}
    if kwargs["uMode"] not in ("min", "max"):
        raise ValueError(f"uMode must be 'min' or 'max', got {kwargs['uMode']!r}")
    return cls(**kwargs)


def make_dynamics(name, params, dim):
    '''
        Full system dynamics by name.
            name: key of DYNAMICS, e.g. "dubins_3d"
            params: dict of constructor arguments, e.g. {"v": 1, "wMax": 1, "uMode": "min"}; uMode defaults
                    to "max" (avoid), dMode to "min", anything else left out takes the class default.
            dim: dimension of the local update, checked against the dynamics
    '''
    check_dynamics(name, dim)
    return _make(DYNAMICS[name]["system"], params or {})


def make_subdynamics(name, params, dim):
    '''
        Subsystem dynamics of the full system name, same parameters as make_dynamics.
        Returns a tuple with one object per subsystem class.
    '''
    check_dynamics(name, dim)
    return tuple(_make(cls, params or {}) for cls in DYNAMICS[name]["subsystems"])
