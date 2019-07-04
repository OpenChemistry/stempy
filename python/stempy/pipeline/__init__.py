import abc
from functools import wraps

class Pipeline(abc.ABC):

    def execute(self, **kwargs):
        """
        Execute pipeline
        """

def pipeline(name=None, description=None):

    def _decorator(func):
        func.NAME = name
        func.DESCRIPTION = description
        func.PARAMETERS = {}
        return func

    return _decorator

def parameter(name, type = 'string', label = None, description = None, default=None):

    def _decorator(func):
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
