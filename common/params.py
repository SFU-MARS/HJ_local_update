# Command-line arguments that can also come from a JSON parameter file (--params).
import json
import os


def parse_args_with_params(parser):
    '''
        Used by main in numpy_impl/local_update_3d.py and hcl_impl/local_update_3d.py. Parse the command line of a parser that has a --params
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
    return parser.parse_args()
