import logging
import typing

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

import determined as det

logging.basicConfig(level=logging.INFO)


class DetCallback(TrainerCallback):
    def __init__(
        self,
        core_context: det.core.Context,
        args: TrainingArguments,
    ) -> None:
        super().__init__()

        self.core_context = core_context
        self.load_last_checkpoint(args)
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
            # Prevents reporting metrics for the same step twice. This happens after
            # training is completed and average training metrics are reported with
            # the same step as the in-progress training metrics.
            if self.last_metrics["train_step"] != state.global_step:
                if state.is_world_process_zero:
                    self.core_context.train.report_training_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                metrics["train_step"] = state.global_step

        elif metric_type == EVAL:
            # Prevents reporting metrics for the same step twice. This happens when
            # after-training evaluation is completed, and it is reported with the same
            # step as the last during-training evaluation.
            if self.last_metrics["eval_step"] != state.global_step:
                if state.is_world_process_zero:
                    self.core_context.train.report_validation_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                metrics["eval_step"] = state.global_step
        else:
            logging.warning(f"Metrics not reported: metric type = {metric_type}.")

        self.last_metrics.update(metrics)

        if self.core_context.preempt.should_preempt():
            control.should_save = True

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        info = det.get_cluster_info()
        assert info

        det_checkpoint_metadata = {
            "steps_completed": state.global_step,
            "trial_id": info.trial.trial_id,
        }

        self.core_context.checkpoint.upload(
            args.output_dir, metadata=det_checkpoint_metadata, shard=True
        )

        if self.core_context.preempt.should_preempt():
            raise Exception("Process preempted / killed")

    def load_last_checkpoint(self, args: TrainingArguments) -> None:
        info = det.get_cluster_info()
        assert info

        latest_checkpoint = info.latest_checkpoint
        if latest_checkpoint is not None:

            self.core_context.checkpoint.download(latest_checkpoint, args.output_dir)
            args.resume_from_checkpoint = args.output_dir

            logging.info(f"Latest checkpoint downloaded to {args.resume_from_checkpoint}.")


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
