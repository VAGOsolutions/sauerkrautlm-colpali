import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)

from sauerkrautlm_colpali.collators import VisualRetrieverCollator
from sauerkrautlm_colpali.data.dataset import ColPaliEngineDataset
from sauerkrautlm_colpali.loss.late_interaction_losses import (
    ColbertLoss,
)
from sauerkrautlm_colpali.trainer.contrastive_trainer import ContrastiveTrainer
from sauerkrautlm_colpali.utils.gpu_stats import print_gpu_utilization, print_summary
from sauerkrautlm_colpali.utils.processing_utils import BaseVisualRetrieverProcessor


@dataclass
class ColModelTrainingConfig:
    model: Union[PreTrainedModel, PeftModel]
    processor: BaseVisualRetrieverProcessor
    train_dataset: Union[ColPaliEngineDataset, List[ColPaliEngineDataset]]
    eval_dataset: Optional[Union[ColPaliEngineDataset, Dict[str, ColPaliEngineDataset]]] = None
    tr_args: Optional[TrainingArguments] = None
    output_dir: Optional[str] = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    train_sampler: Optional[Any] = None  # Custom batch sampler (z.B. GroupedBatchSampler)
    loss_func: Optional[Callable] = ColbertLoss()
    pretrained_peft_model_name_or_path: Optional[str] = None
    """
    Config class used for training a ColVision model.
    """

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        if self.output_dir is None:
            sanitized_name = str(self.model.name_or_path).replace("/", "_")
            self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            print("No training arguments provided. Using default.")
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        elif self.tr_args.output_dir is None or self.tr_args.output_dir == "trainer_output":
            self.tr_args.output_dir = self.output_dir

        if isinstance(self.tr_args.learning_rate, str):
            print("Casting learning rate to float")
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)

        self.tr_args.remove_unused_columns = False

        if self.pretrained_peft_model_name_or_path is not None:
            print("Loading pretrained PEFT model")
            self.model.load_adapter(self.pretrained_peft_model_name_or_path, is_trainable=True)

        if self.peft_config is not None:
            print("Configurating PEFT model")
            if self.pretrained_peft_model_name_or_path is None:
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()
            else:
                print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting.")

    print_gpu_utilization()


class ColModelTraining:
    """
    Class that contains the training and evaluation logic for a ColVision model.
    """

    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        self.train_dataset = self.config.train_dataset
        self.eval_dataset = self.config.eval_dataset
        self.collator = VisualRetrieverCollator(
            processor=self.config.processor,
            max_length=self.config.max_length,
        )

    def train(self, custom_trainer_class=None, custom_trainer_kwargs=None) -> None:
        """
        Start training.
        
        Args:
            custom_trainer_class: Optional custom trainer class to use instead of ContrastiveTrainer
            custom_trainer_kwargs: Optional kwargs to pass to custom trainer
        """
        if custom_trainer_class is not None:
            # Use custom trainer (e.g., for MTEB evaluation)
            trainer_kwargs = {
                "model": self.model,
                "train_dataset": self.train_dataset,
                "eval_dataset": self.eval_dataset,
                "args": self.config.tr_args,
                "data_collator": self.collator,
                "loss_func": self.config.loss_func,
                "is_vision_model": self.config.processor is not None,
            }
            if custom_trainer_kwargs:
                trainer_kwargs.update(custom_trainer_kwargs)
            trainer = custom_trainer_class(**trainer_kwargs)
        else:
            trainer = ContrastiveTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                args=self.config.tr_args,
                data_collator=self.collator,
                loss_func=self.config.loss_func,
                is_vision_model=self.config.processor is not None,
            )

        trainer.args.remove_unused_columns = False
        
        # Set custom sampler if provided
        if self.config.train_sampler is not None:
            print(f"   ✅ Using custom train_sampler: {type(self.config.train_sampler).__name__}")
            trainer._train_sampler = self.config.train_sampler

        result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint)
        print_summary(result)

    def eval(self) -> None:
        raise NotImplementedError("Evaluation is not implemented yet.")

    def save(self):
        """
        Save the model with its training config, as well as the tokenizer and processor if provided.
        """
        self.model.save_pretrained(self.config.output_dir)
        self.config.processor.save_pretrained(self.config.output_dir)

        # Save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
