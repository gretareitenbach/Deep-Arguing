import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, TypeVar, Union, override

import wandb

T = TypeVar("T")


def _retry_with_backoff(
    func: Callable[[], T],
    operation_name: str,
    max_retries: int = 5,
    delays: tuple[int, ...] = (2, 6, 18, 54, 162),
) -> tuple[T | None, bool]:
    """
    Retry a function with exponential backoff.

    Parameters
    ----------
    func : Callable[[], T]
        The function to execute.
    operation_name : str
        Name of the operation for logging purposes.
    max_retries : int
        Maximum number of retry attempts.
    delays : tuple[int, ...]
        Delay in seconds before each retry attempt.

    Returns
    -------
    tuple[T | None, bool]
        A tuple of (result, success). If all retries fail, returns (None, False).
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            result = func()
            return (result, True)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = delays[attempt] if attempt < len(delays) else delays[-1]
                logging.warning(
                    f"W&B {operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                logging.warning(
                    f"W&B {operation_name} failed after {max_retries} attempts: {last_exception}. "
                    f"Falling back to dummy logger for this trial."
                )

    return (None, False)


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
    def init(self, project: str, config: Dict[str, Any] | None = None, group: str = "") -> None:
        """Initialize the logger (e.g. start a new run)."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_metrics(
        self, metrics: Dict[str, Union[int, float]], step: int | None = None
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

    Includes retry logic with exponential backoff for network failures.
    If all retries fail, falls back to dummy logger behavior for the
    remainder of the trial, then tries W&B again on the next trial.
    """

    def __init__(self) -> None:
        self.run: Any = None
        self._fallback_mode: bool = False

    @override
    def init(self, project: str, config: Dict[str, Any] | None = None, group: str = "") -> None:
        # Reset fallback mode at the start of each trial
        self._fallback_mode = False

        def _init() -> Any:
            run = wandb.init(project=project, config=config, group=group)
            if run is None:  # pyright: ignore[reportUnnecessaryComparison]
                raise Exception("wandb.init returned None")
            return run

        result, success = _retry_with_backoff(_init, "init")
        if success and result is not None:
            self.run = result
        else:
            self._fallback_mode = True
            self.run = None

    @override
    def log_params(self, params: Dict[str, Any]) -> None:
        if self._fallback_mode:
            return

        def _log_params() -> None:
            wandb.config.update(params)

        _, success = _retry_with_backoff(_log_params, "log_params")
        if not success:
            self._fallback_mode = True

    @override
    def log_metrics(
        self, metrics: Dict[str, Union[int, float]], step: int | None = None
    ) -> None:
        if self._fallback_mode:
            return

        def _log_metrics() -> None:
            wandb.log(metrics, step=step)

        _, success = _retry_with_backoff(_log_metrics, "log_metrics")
        if not success:
            self._fallback_mode = True

    @override
    def log_artifact(self, name: str, path: str, type: str = "dataset") -> None:
        if self._fallback_mode:
            return

        def _log_artifact() -> None:
            if self.run is None:
                raise Exception("Cannot log artifact: run is None")
            artifact = wandb.Artifact(name, type=type)
            artifact.add_file(path)
            self.run.log_artifact(artifact)

        _, success = _retry_with_backoff(_log_artifact, "log_artifact")
        if not success:
            self._fallback_mode = True

    @override
    def finish(self) -> None:
        if self._fallback_mode or self.run is None:
            # Reset for next trial
            self._fallback_mode = False
            return

        def _finish() -> None:
            if self.run is not None:
                self.run.finish()

        _, success = _retry_with_backoff(_finish, "finish")
        # Always reset fallback mode for next trial, regardless of success
        self._fallback_mode = False
        if not success:
            # If finish failed, still clear the run reference
            self.run = None

class DummyLogger(ExperimentLogger):

    def __init__(self) -> None:
        self.run = None

    @override
    def init(self, project: str, config: Dict[str, Any] | None = None, group: str = "") -> None:
        pass

    @override
    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    @override
    def log_metrics(
        self, metrics: Dict[str, Union[int, float]], step: int | None = None
    ) -> None:
        pass

    @override
    def log_artifact(self, name: str, path: str, type: str = "dataset") -> None:
        pass

    @override
    def finish(self) -> None:
        pass
