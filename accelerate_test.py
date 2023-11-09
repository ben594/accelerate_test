from accelerate import Accelerator, DistributedType
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
import evaluate

accelerator = Accelerator()
EVAL_BATCH_SIZE = 16
MAX_GPU_BATCH_SIZE = 16

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
            remove_colums = ["idx", "sentence1", "sentence2"]
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