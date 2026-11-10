# How the two subsystem value functions are combined into the full value function:
#   union:        V = min(V1, V2)
#   intersection: V = max(V1, V2)
COMBINE_CHOICES = ("union", "intersection")


def check_combine(combine, supported, case):
    '''
        Raise an error if combine is not a known setting, or not implemented for this case.
            combine: "union" or "intersection"
            supported: the settings this case implements, e.g. ("union",)
            case: name used in the error message, e.g. "3d"
    '''
    if combine not in COMBINE_CHOICES:
        raise ValueError(f"combine must be one of {list(COMBINE_CHOICES)}, got {combine!r}")
    if combine not in supported:
        raise NotImplementedError(f"combine='{combine}' is not implemented for {case}, "
                                  f"only {', '.join(supported)}")
