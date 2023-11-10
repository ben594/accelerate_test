from accelerate import Accelerator, DistributedType
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
import evaluate
import argparse

EVAL_BATCH_SIZE = 16
MAX_GPU_BATCH_SIZE = 12

def get_dataloaders(accelerator, batch_size = 16):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    datasets = load_dataset("glue", "mrpc")
    
    def tokenize(examples):
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], examples["sentence3"])
        return outputs
    
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize,
            batched = True,
            remove_columns = ["idx", "sentence1", "sentence2"]
        )
        
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    def collate_fn(examples):
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding = "longest",
            max_length = None,
            pad_to_multiple_of = pad_to_multiple_of,
            return_tensors = "pt",
        )
        
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle = True,
        collate_fn = collate_fn,
        batch_size = batch_size,
        drop_last = True
    )
    
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle = False,
        collate_fn = collate_fn,
        batch_size = EVAL_BATCH_SIZE,
        drop_last = (accelerator.mixed_precision == "fp8")
    )
    
    return train_dataloader, eval_dataloader

def training_function(config, args):
    accelerator = Accelerator(cpu = args.cpu, mixed_precision = args.mixed_precision)
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    
    metric = evaluate.load("glue", "mrpc")
    
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_stteps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE
        
    set_seed(seed)
    train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", return_dict=True)
    
    model = model.to(accelerator.device)
    
    optimizer = AdamW(params=model.parameters(), lr=lr)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * num_epochs) // gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # train model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)
        
def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    training_function(config, args)


if __name__ == "__main__":
    main()