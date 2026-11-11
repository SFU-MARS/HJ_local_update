# Command-line arguments that can also come from a JSON parameter file (--params).
import json
import os


def parse_args_with_params(parser):
    '''
        Used by main of the local update scripts. Parse the command line of a parser that has a --params
        argument. The keys of the JSON file given by --params (argument names without '--') replace the
        defaults, so arguments given on the command line still override the file.
        Returns the parsed arguments.
    '''
    params_path = parser.parse_known_args()[0].params
    if params_path is not None:
        if not os.path.isfile(params_path):
            parser.error(f"file not found: {params_path}")
        with open(params_path) as f:
            params = json.load(f)
        actions = {action.dest: action for action in parser._actions}
        unknown = sorted(set(params) - set(actions) - {"params"})
        if unknown:
            parser.error(f"unknown keys in {params_path}: {unknown}")
        for key, value in params.items():
            action = actions[key]
            if value is not None and action.type is not None:
                params[key] = action.type(value)
            if action.choices is not None and params[key] not in action.choices:
                parser.error(f"{key} in {params_path} must be one of {list(action.choices)}, got {value!r}")
        parser.set_defaults(**params)
    args = parser.parse_args()
    # "null" on the command line means not given, like null in the JSON file (e.g. --true_file null)
    for key, value in vars(args).items():
        if value == "null":
            setattr(args, key, None)
    return args


def json_object(value):
    '''
        argparse type for a JSON object: a dict from a --params file, or a JSON string on the command
        line, e.g. --dynamics_params '{"v": 1, "wMax": 1}'.
    '''
    if isinstance(value, dict):
        return value
    try:
        obj = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {value!r} ({e})")
    if not isinstance(obj, dict):
        raise ValueError(f"must be a JSON object like {{\"v\": 1}}, got {value!r}")
    return obj


def add_compute_args(parser, grid_size, combine, t_step, lookback_length, dynamics):
    '''
        Arguments of the compute scripts (compute/direct_<dim>.py, compute/decomposition_<dim>.py), the same
        for every case, only the defaults differ. Use the same values for the direct computation and the
        decomposition; the local update reads the grid size and number of time steps from the saved files.
    '''
    from dynamics import add_dynamics_args
    parser.add_argument("--grid_size", type=int, default=grid_size, help=f"Grid points per dimension (default {grid_size}).")
    parser.add_argument("--combine", choices=["union", "intersection"], default=combine,
                        help="union = min(V1, V2) with uMode max (avoid), intersection = max(V1, V2) with uMode min (reach).")
    parser.add_argument("--t_step", type=float, default=t_step, help="Time step.")
    parser.add_argument("--lookback_length", type=float, default=lookback_length,
                        help="Look-back length, lookback_length / t_step time steps.")
    parser.add_argument("--results_dir", default="results", help="Directory of the saved results.")
    add_dynamics_args(parser, default=dynamics)
