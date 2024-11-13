# fine_tuning.py
from transformers import Trainer, TrainingArguments

def fine_tune_model(model, dataset_path, tokenizer):
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=dataset_path)["train"]

    def preprocess_function(examples):
        inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=256)
        labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=256).input_ids
        inputs["labels"] = labels
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    training_args = TrainingArguments(output_dir="./results", per_device_train_batch_size=4, num_train_epochs=3)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()
    return model
