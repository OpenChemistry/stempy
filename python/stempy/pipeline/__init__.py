import abc


class Pipeline(abc.ABC):

    def execute(self, **kwargs):
        """
        Execute pipeline
        """

def pipeline(name=None, description=None):

    def _decorator(func):
        func.NAME = name
        func.DESCRIPTION = description

        return func

    return _decorator
