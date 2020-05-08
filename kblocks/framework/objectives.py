from typing import Any, Dict, Optional

import gin


@gin.configurable(module="kb.framework")
class Objective(object):
    def __init__(self, name: str, split="validation", mode: str = "max"):
        self._name = name
        self._mode = mode
        self._split = split

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def split(self):
        return self._split

    def get_config(self) -> Dict:
        return dict(name=self.name, mode=self.mode, split=self.split)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        return Objective(**config)

    @classmethod
    def get(cls, identifier) -> Optional["Objective"]:
        if identifier is None:
            return identifier
        elif isinstance(identifier, Objective):
            return identifier
        if isinstance(identifier, (list, tuple)):
            return Objective(*identifier)
        elif isinstance(identifier, dict):
            return Objective(**identifier)
        elif isinstance(identifier, str):
            return Objective(identifier)
        else:
            raise TypeError(
                "Cannot convert identifier {} into an Objective".format(identifier)
            )
