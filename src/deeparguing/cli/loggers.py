from abc import ABC, abstractmethod
from typing import Any, Dict, Union, override

import wandb


class ExperimentLogger(ABC):
    """
    Abstract base class for logging ML experiments.


    """

    _current = None

    @classmethod
    def set_current(cls, logger: "ExperimentLogger") -> None:
        cls._current = logger

    @classmethod
    def current(cls) -> "ExperimentLogger":
        if cls._current is None:
            raise RuntimeError("No current logger set!")
        return cls._current

    @abstractmethod
    def init(self, project: str, config: Dict[str, Any] = None, group: str = "") -> None:
        """Initialize the logger (e.g. start a new run)."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, Union[int, float]], step: int = None
    ) -> None:
        """Log metrics (e.g. loss, accuracy)."""
        pass

    @abstractmethod
    def log_artifact(self, name: str, path: str, type: str = "dataset") -> None:
        """Log an artifact (file, dataset, model weights, etc.)."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finalize/close the logger."""
        pass


class WandbLogger(ExperimentLogger):
    """
    Weights & Biases implementation of ExperimentLogger.
    """

    def __init__(self):
        self.run = None

    @override
    def init(self, project: str, config: Dict[str, Any] = None, group: str = "") -> None:
        self.run = wandb.init(project=project, config=config, group=group)

    @override
    def log_params(self, params: Dict[str, Any]) -> None:
        wandb.config.update(params)

    @override
    def log_metrics(
        self, metrics: Dict[str, Union[int, float]], step: int = None
    ) -> None:
        wandb.log(metrics, step=step)

    @override
    def log_artifact(self, name: str, path: str, type: str = "dataset") -> None:
        artifact = wandb.Artifact(name, type=type)
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    @override
    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

class DummyLogger(ExperimentLogger):

    def __init__(self):
        self.run = None

    @override
    def init(self, project: str, config: Dict[str, Any] = None, group: str = "") -> None:
        pass

    @override
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    @override
    def log_metrics(
        self, metrics: Dict[str, Union[int, float]], step: int = None
    ) -> None:
        pass

    @override
    def log_artifact(self, name: str, path: str, type: str = "dataset") -> None:
        pass

    @override
    def finish(self) -> None:
        pass
