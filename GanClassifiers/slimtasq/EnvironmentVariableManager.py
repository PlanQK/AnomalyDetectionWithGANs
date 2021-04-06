import os
import typing


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kw):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kw)
        return cls._instances[cls]


class EnvironmentVariableManager(metaclass=Singleton):
    """Singleton class to access environment variables. On first
    initialization default values can be specified in the constructor call.
    """

    def __init__(self, defaultVariables: typing.Dict[str, typing.Any] = {}):
        self.envVariables = defaultVariables.copy()
        self.envVariables.update(self.returnEnvironmentVariables())

    @staticmethod
    def returnEnvironmentVariables() -> typing.Dict[str, typing.Any]:
        return os.environ

    def __getitem__(self, name: str) -> typing.Any:
        return self.envVariables.get(name, None)
