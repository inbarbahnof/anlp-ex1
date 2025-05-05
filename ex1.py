from transformers import (AutoTokenizer, TrainerCallback,
                          Trainer, TrainingArguments, AutoModelForSequenceClassification,
                          HfArgumentParser)
from evaluate import load
import numpy as np
import wandb
from dataclasses import dataclass, field
from datasets import load_dataset


@dataclass
class Arguments:
    max_train_samples: int = field(
        default=-1,
        metadata={"help": "Number of samples to use for training. Use -1 for all."}
    )
    max_eval_samples: int = field(
        default=-1,
        metadata={"help": "Number of samples to use for evaluation. Use -1 for all."}
    )
    max_predict_samples: int = field(
        default=-1,
        metadata={"help": "Number of samples to use for prediction. Use -1 for all."}
    )
    model_path: str = field(
        default="bert-base-uncased",
        metadata={"help": "Path to a pretrained model (for prediction or training)"}
    )
    lr: float = field(
            default=5e-5,
            metadata={"help": "Learning rate"}
    )
    batch_size: int = field(
            default=16,
            metadata={"help": "Training batch size"}
    )


def preprocess_text(dataset):
    return tokenizer(
        dataset["sentence1"],
        dataset["sentence2"],
        truncation=True,
        padding="longest",
        max_length=512,
        return_tensors="pt"
    )


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    acc = accuracy_metric.compute(predictions=preds, references=p.label_ids)
    return {"eval_accuracy": acc}


class LossLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history and "loss" in state.log_history[-1]:
            loss = state.log_history[-1]["loss"]
            wandb.log({"train/loss": loss, "step": state.global_step})


if __name__ == '__main__':
    wandb.login()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # create tokenizer
    accuracy_metric = load("accuracy")  # create accuracy_metric

    raw_datasets = load_dataset("glue", "mrpc")  # load dataset
    processed_datasets = raw_datasets.map(preprocess_text, batched=True)  # process dataset

    # load arguments
    parser = HfArgumentParser((Arguments, TrainingArguments))
    custom_args, training_args = parser.parse_args_into_dataclasses()

    training_args.learning_rate = custom_args.lr
    training_args.per_device_train_batch_size = custom_args.batch_size

    run_name = f"mrpc-e{training_args.num_train_epochs}-lr{custom_args.lr}-bs{custom_args.batch_size}"
    wandb.init(project="anlp-ex1", name=run_name)

    # create a model
    model = AutoModelForSequenceClassification.from_pretrained(custom_args.model_path)

    # save datasets
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    # change costume arguments info
    if custom_args.max_train_samples != -1:
        train_dataset = train_dataset.select(range(custom_args.max_train_samples))

    if custom_args.max_eval_samples != -1:
        eval_dataset = eval_dataset.select(range(custom_args.max_eval_samples))

    if custom_args.max_predict_samples != -1:
        test_dataset = test_dataset.select(range(custom_args.max_predict_samples))

    # create a trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[LossLoggerCallback()],
    )

    if training_args.do_train:
        trainer.train()
        trainer.model.save_pretrained(f"./results/{run_name}/final_model")
        tokenizer.save_pretrained(f"./results/{run_name}/final_model")

    if training_args.do_predict:
        model = AutoModelForSequenceClassification.from_pretrained(
            custom_args.model_path,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            custom_args.model_path,
            local_files_only=True
        )
        trainer.model = model  # ensure Trainer uses the loaded model
        model.eval()

        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)

        pred_path = "predictions.txt"
        with open(pred_path, "w") as f:
            for i, label in enumerate(pred_labels):
                s1 = raw_datasets["test"]["sentence1"][i]
                s2 = raw_datasets["test"]["sentence2"][i]
                f.write(f"<{s1}>###<{s2}>###<{label}>\n")
