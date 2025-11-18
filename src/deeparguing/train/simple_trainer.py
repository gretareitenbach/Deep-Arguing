from typing import Any, Callable, override

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from deeparguing import GradualAACBR
from deeparguing.cli.loggers import ExperimentLogger
from deeparguing.regulariser import RegulariserType
from deeparguing.train import Trainer

from torch.profiler import record_function


class SimpleTrainer(Trainer):

    def __init__(
        self,
        real_time_logger: Callable[[Any], Any] = lambda _: None,
    ) -> None:
        super().__init__(real_time_logger)

    @override
    def train(
        self,
        model: GradualAACBR,
        X_casebase: Tensor,
        y_casebase: Tensor,
        X_new_cases: Tensor,
        y_new_cases: Tensor,
        X_default: Tensor,
        y_default: Tensor,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        epochs: int,
        regulariser: RegulariserType = lambda _: 0,
        disable_tqdm: bool = False,
        batch_size: None | int = None,
        scheduler: LRScheduler | None = None,
        gradient_max_norm: float | None = None,
        X_val: Tensor | None = None,
        y_val: Tensor | None = None,
        log_val_loss: bool = False,
    ):

        pbar = tqdm(range(epochs), dynamic_ncols=True, disable=disable_tqdm)

        n_samples = X_new_cases.shape[0]

        batch_size = batch_size if batch_size is not None else n_samples
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # since with_stack=True caused segfault
        ) as prof:
            for epoch in pbar:
                permutation = torch.randperm(n_samples, device=model.device)

                for i in range(0, n_samples, batch_size):
                    with record_function("my_index_batch"):
                        indices = permutation[i : i + batch_size]
                        batch_X_new_cases = X_new_cases[indices]
                        batch_y_new_cases = y_new_cases[indices]
                    with record_function("my_train_step"):
                       loss = self._train_step(
                           model,
                           X_casebase,
                           y_casebase,
                           batch_X_new_cases,
                           batch_y_new_cases,
                           X_default,
                           y_default,
                           optimizer,
                           criterion,
                           regulariser=regulariser,
                           scheduler=scheduler,
                           gradient_max_norm=gradient_max_norm,
                       )

                prof.step()
            # assert loss

            # pbar.set_description(f"Epoch {epoch + 1}, Loss: {round(loss.item(), 6)}")

            # if log_val_loss and X_val is not None and y_val is not None:
            #     val_loss_avg = self.log_validation_loss(
            #         model, batch_size, X_val, y_val, criterion, regulariser
            #     )
            #     ExperimentLogger.current().log_metrics(
            #         {
            #             "loss_per_epoch": float(loss.item()),
            #             "epoch": epoch,
            #             "val_loss_per_epoch": float(val_loss_avg),
            #         }
            #     )
            # else:
            #     ExperimentLogger.current().log_metrics(
            #         {"loss_per_epoch": float(loss.item()), "epoch": epoch}
            #     )
