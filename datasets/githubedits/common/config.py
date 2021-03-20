import json
from collections import OrderedDict


def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def isint(x):
    try:
        if '.' in x:
            return False

        a = float(x)
        b = int(a)
    except:
        return False
    else:
        return a == b


class Arguments(OrderedDict):
    EVAL_ARGS = {'--gnn_layer_timesteps', '--gnn_residual_connections', '--gnn_connections'}

    def __init__(self, *args, **kwargs):
        super(Arguments, self).__init__(*args, **kwargs)

    @staticmethod
    def from_file(file_path, cmd_args=None):
        config = json.load(open(file_path, 'r'))
        
        args = Arguments(config)
        args['cmd_args'] = cmd_args

        return args

    def to_string(self):
        return json.dumps(self, indent=2)
