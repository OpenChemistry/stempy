import abc
from functools import wraps
from collections import OrderedDict

class PipelineIO:
    Image = 'image'
    Frame = 'frame'

class PipelineAggregation:
    Sum = 'sum'
    Max = 'max'
    Min = 'min'
    Mean = 'mean'
    Median = 'median'

class Pipeline(abc.ABC):

    def execute(self, files, **kwargs):
        """
        Execute pipeline
        """

def _ensure_parameters(f):
    if not hasattr(f, 'PARAMETERS'):
        setattr(f, 'PARAMETERS', OrderedDict())

def pipeline(name=None, description=None,
             input=PipelineIO.Frame, output=PipelineIO.Image,
             aggregation=PipelineAggregation.Sum):

    def _decorator(func):
        func.NAME = name
        func.DESCRIPTION = description
        func.INPUT = input
        func.OUTPUT = output
        func.AGGREGATION = aggregation
        _ensure_parameters(func)
        return func

    return _decorator

def parameter(name, type = 'string', label = None, description = None, default=None):

    def _decorator(func):
        _ensure_parameters(func)
        func.PARAMETERS[name] = {
            'type': type,
            'label': label,
            'description': description,
            'default': default
        }

        @wraps(func)
        def _wrapper(*args, **kwargs):
            if name not in kwargs:
                kwargs[name] = default
            return func(*args, **kwargs)
        return _wrapper

    return _decorator
