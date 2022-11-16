import typing

from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
)

import os
import determined as det
import logging


class DetCallback(TrainerCallback):
    def __init__(self, core_context: det.core.Context, args: TrainingArguments) -> None:
        super().__init__()

        self.core_context = core_context
        self.load_last_checkpoint(args)

        self.last_metrics: typing.Dict[str, float] = {'train_step': 0, 'eval_step': 0}

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):

        if state.is_world_process_zero:
            metrics = logs
            metric_type = get_metric_type(logs)

            if metric_type == TRAIN and state.global_step != self.last_metrics['train_step']:
                self.core_context.train.report_training_metrics(steps_completed=state.global_step, metrics=metrics)
                metrics['train_step'] = state.global_step
            elif metric_type == EVAL and state.global_step != self.last_metrics['eval_step']:
                self.core_context.train.report_validation_metrics(steps_completed=state.global_step, metrics=metrics)
                metrics['eval_step'] = state.global_step
            else:
                logging.warning(f"Metrics not reported: metric type = {metric_type}.")

            self.last_metrics.update(metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs, ):
        info = det.get_cluster_info()
        local_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        if state.is_world_process_zero:
            det_checkpoint_metadata = {
                "steps_completed": state.global_step,
                "trial_id": info.trial.trial_id,
            }

            ### save your stuff here
            with open(os.path.join(local_path, 'hello_there.txt'), 'w') as f:
                f.write('we have the high ground')

            storage_id = self.core_context.checkpoint.upload(local_path, metadata=det_checkpoint_metadata)
            logging.info(f'storage_id={storage_id}')

        if self.core_context.preempt.should_preempt():
            raise Exception("Process preempted / killed")

    def load_last_checkpoint(self, args: TrainingArguments) -> None:
        info = det.get_cluster_info()
        latest_checkpoint = info.latest_checkpoint

        logging.info(f'latest_checkpoint={latest_checkpoint}')
        if latest_checkpoint is not None:
            metadata = self.core_context.checkpoint.get_metadata(latest_checkpoint)
            prev_trial_id = metadata["trial_id"]
            trial_id = info.trial.trial_id
            if trial_id != prev_trial_id:
                # WebUI: Continue Trial - starts a new trial, using the checkpoint weights
                # for initialization but starting training from batch 0.
                resume_step = 0
            else:
                # WebUI: Pause/Resume - resume training where we left off.
                resume_step = metadata["steps_completed"]
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{resume_step}")

            self.core_context.checkpoint.download(latest_checkpoint, checkpoint_path)

            args.resume_from_checkpoint = checkpoint_path

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs, ):
        if self.core_context.preempt.should_preempt():
            control.should_save = True


EVAL = "eval_"
TRAIN_AVG = "train_"
TRAIN = "train_progress"

'''
Formats of logs:
1) During training: 
   {'loss': 0.7902, 'learning_rate': 0.0, 'epoch': 0.77}
2) During eval:
   {'eval_loss': 0.7956713438034058, 'eval_accuracy': 0.9172932330827067, 'eval_runtime': 2.0982, 'eval_samples_per_second': 63.386, 'eval_steps_per_second': 8.102, 'epoch': 0.77}
3) Averaged training:
   {'train_runtime': 104.0137, 'train_samples_per_second': 7.691, 'train_steps_per_second': 0.961, 'train_loss': 0.9000937604904174, 'epoch': 0.77}
'''


def get_metric_type(d):
    for k, v in d.items():
        if k.startswith(EVAL):
            return EVAL
        elif k.startswith(TRAIN_AVG):
            return TRAIN_AVG
        else:
            return TRAIN
