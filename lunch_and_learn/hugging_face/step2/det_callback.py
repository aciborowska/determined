import logging
import typing

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import determined as det

logging.basicConfig(level=logging.INFO)


class DetCallback(TrainerCallback):
    def __init__(self, core_context: det.core.Context) -> None:
        super().__init__()

        self.core_context = core_context
        self.last_metrics: typing.Dict[str, float] = {"train_step": -1, "eval_step": -1}

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        metrics = logs
        metric_type = get_metric_type(logs)

        if metric_type == TRAIN:
            if self.last_metrics["train_step"] != state.global_step:
                if state.is_world_process_zero:
                    self.core_context.train.report_training_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                metrics["train_step"] = state.global_step

        elif metric_type == EVAL:
            if self.last_metrics["eval_step"] != state.global_step:
                if state.is_world_process_zero:
                    self.core_context.train.report_validation_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                metrics["eval_step"] = state.global_step
        else:
            logging.warning(f"Metrics not reported: metric type = {metric_type}.")

        self.last_metrics.update(metrics)


EVAL = "eval_"
TEST = "test_"
TRAIN_AVG = "train_"
TRAIN = "train_progress"


def get_metric_type(d):
    for k, v in d.items():
        if k.startswith(EVAL):
            return EVAL
        elif k.startswith(TEST):
            return TEST
        elif k.startswith(TRAIN_AVG):
            return TRAIN_AVG
        else:
            return TRAIN
