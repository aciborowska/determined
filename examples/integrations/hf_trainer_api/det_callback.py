import json
import logging
import os
import typing

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

import determined as det


class DetCallback(TrainerCallback):
    def __init__(
        self,
        core_context: det.core.Context,
        args: TrainingArguments,
        filter_metrics: typing.List[str] = None,
        user_data: typing.Dict = None,
    ) -> None:
        super().__init__()

        self.core_context = core_context

        self.filter_metrics = filter_metrics
        self.user_data = user_data
        self.load_last_checkpoint(args)

        self.last_metrics: typing.Dict[str, float] = {"train_step": -1, "eval_step": -1}

        searcher_config = det.get_cluster_info().trial._config["searcher"]
        self.searcher_metric = searcher_config["metric"]
        self.searcher_unit = list(searcher_config["max_length"].keys())[0]
        self.searcher_max_length = list(searcher_config["max_length"].values())[0]
        self.searcher_ops = self.core_context.searcher.operations()
        self.current_op = next(self.searcher_ops)
        self._check_searcher_compatibility(args)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if state.is_world_process_zero:
            metrics, metric_type = self._get_metrics(logs)
            if metric_type == TRAIN:
                # Prevents reporting metrics for the same step twice. This happens after
                # training is completed and average training metrics are reported with
                # the same step as the in-progress training metrics.
                if self.last_metrics["train_step"] != state.global_step:
                    self.core_context.train.report_training_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                    metrics["train_step"] = state.global_step

            elif metric_type == EVAL:
                # Prevents reporting metrics for the same step twice. This happens when
                # after-training evaluation is completed, and it is reported with the same
                # step as the last during-training evaluation.
                if self.last_metrics["eval_step"] != state.global_step:
                    self.core_context.train.report_validation_metrics(
                        steps_completed=state.global_step, metrics=metrics
                    )
                    metrics["eval_step"] = state.global_step
                    self.last_metrics.update(metrics)
            else:
                logging.warning(f"Metrics not reported: metric type = {metric_type}.")

            self.last_metrics.update(metrics)

        if self.core_context.preempt.should_preempt():
            control.should_save = True

    def _get_metrics(self, logs: typing.Dict) -> typing.Tuple[typing.Dict, str]:
        metrics = logs
        metric_type = get_metric_type(logs)
        if self.filter_metrics:
            metrics = {}
            for k, v in logs.items():
                if any(m in k for m in self.filter_metrics) is True:
                    metrics[k] = v

        return metrics, metric_type

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        info = det.get_cluster_info()
        assert info

        # local_path is where HF Trainer saves model and tokenizer in a given step.
        local_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if state.is_world_process_zero:
            if self.user_data is not None:
                self._on_save_user_data(local_path)

        det_checkpoint_metadata = {
            "steps_completed": state.global_step,
            "trial_id": info.trial.trial_id,
        }

        def selector(x: str) -> bool:
            return x.startswith((f"checkpoint-{state.global_step}/", "runs/"))

        self.core_context.checkpoint.upload(
            args.output_dir, metadata=det_checkpoint_metadata, shard=True, selector=selector
        )

        if self.core_context.preempt.should_preempt():
            raise Exception("Process preempted / killed")

    def _on_save_user_data(self, save_path: str) -> None:
        """
        User-defined saving of objects from self.checkpoint_metadata under save_path.
        After objects are saved, Determined handles uploading and downloading objects to/from selected storage.
        """
        with open(os.path.join(save_path, "my_data.json"), "w") as f:
            json.dump(self.user_data, f)

    def load_last_checkpoint(self, args: TrainingArguments) -> None:
        info = det.get_cluster_info()
        assert info

        latest_checkpoint = info.latest_checkpoint
        if latest_checkpoint is not None:

            # To resume DeepSpeed, each node requires ALL sharded model/optimizer states,
            # so we can skip using selector and just download all files.
            self.core_context.checkpoint.download(latest_checkpoint, args.output_dir)

            # Use metadata to decide whether the current trial is forked/continued
            # (trial_id != prev_trial_id) or unpaused (trial_id == prev_trial_id).
            # Set checkpoint path for Trainer to load the last reported checkpoint.
            metadata = self.core_context.checkpoint.get_metadata(latest_checkpoint)
            prev_trial_id = metadata["trial_id"]
            trial_id = info.trial.trial_id
            if trial_id != prev_trial_id:
                resume_step = 0
            else:
                resume_step = metadata["steps_completed"]

            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{resume_step}")
            args.resume_from_checkpoint = checkpoint_path

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # state.epoch is not None only during training.
        if state.epoch and self.searcher_unit == "batches":
            if state.is_world_process_zero:
                self.current_op.report_progress(state.global_step)

            if state.global_step >= self.current_op.length:
                self._update_searcher(state, control)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # state.epoch is not None only during training.
        if state.epoch and self.searcher_unit == "epochs":
            if state.is_world_process_zero:
                self.current_op.report_progress(state.epoch)

            if round(state.epoch) >= self.current_op.length:
                self._update_searcher(state, control)

    def _update_searcher(self, state: TrainerState, control: TrainerControl) -> None:
        if state.is_world_process_zero:
            if self.last_metrics is None:
                logging.warning(
                    f"No training or evaluation metrics has been recorded. Please check your settings for "
                    f"training metrics (--logging_strategy and --logging_steps) or "
                    f"evaluation metrics (--evaluation_strategy and --eval_steps). "
                    f"Reporting trainer_state.best_metric to the searcher."
                )
                self.current_op.report_completed(state.best_metric)
            elif self.searcher_metric not in self.last_metrics:
                logging.warning(
                    f"Searcher metric {self.searcher_metric} from the yaml config file does not match any "
                    f"of the recorded metrics in {self.last_metrics}. "
                    f"Reporting trainer_state.best_metric to the searcher."
                )
                self.current_op.report_completed(state.best_metric)
            else:
                self.current_op.report_completed(self.last_metrics[self.searcher_metric])

        try:
            self.current_op = next(self.searcher_ops)
        except StopIteration:
            control.should_training_stop = True

    def _check_searcher_compatibility(self, args: TrainingArguments) -> None:
        if self.searcher_unit == "batches":
            if args.max_steps == -1:
                self._log_config_mismatch("epochs", args.num_train_epochs)
            elif args.max_steps != self.searcher_max_length:
                self._log_config_mismatch("batches", args.max_steps)
        elif self.searcher_unit == "epochs":
            if args.max_steps != -1:
                self._log_config_mismatch("batches", args.max_steps)
            elif args.num_train_epochs != self.searcher_max_length:
                self._log_config_mismatch("epochs", args.num_train_epochs)

    def _log_config_mismatch(
        self,
        trainer_units: str,
        trainer_len: float,
    ) -> None:
        logging.warning(
            f"Searcher configuration does not match HF Trainer configuration. "
            f"Searcher uses {self.searcher_unit}={self.searcher_max_length}, "
            f"while HF Trainer uses {trainer_units}={trainer_len}. "
            f"Continuing this run may cause Searcher not to behave correctly. "
            f"Make sure to match the units between HF Trainer and Searcher: "
            f"use (--num_train_epochs and searcher.max_length.epochs) OR "
            f"(--max_steps and searcher.max_length.batches)."
        )


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
