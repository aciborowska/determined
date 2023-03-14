from transformers.trainer import TrainerCallback, TrainerControl, TrainingArguments, TrainerState
import logging
import determined as det

logging.basicConfig(level=logging.INFO)


# What is the content of logs reported by Trainer API:
# during training: {'loss': 4.0286, 'learning_rate': 9.94e-05, 'epoch': 0.03}
# training summary: {'train_runtime': 195.1185, 'train_samples_per_second': 4.1, 'train_steps_per_second': 0.513, 'train_loss': 4.509950866699219, 'epoch': 0.12}
# in eval: {'eval_loss': 3.5561270713806152, 'eval_accuracy': 0.49278423090461104, 'eval_runtime': 78.614, 'eval_samples_per_second': 72.277, 'eval_steps_per_second': 4.528, 'epoch': 0.05}
