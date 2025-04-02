import os
import random
import torch
import torch.distributed as dist
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import re
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# --------------------------------
# Distributed Training Setup for ROCm
# --------------------------------
def init_distributed():
    """Initialize Distributed Training (assumes ROCm‑compatible PyTorch build)."""
    world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS', 1)))
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', 0)))
    
    # For ROCm builds of PyTorch the CUDA interface is still used.
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Use NCCL backend, which works with ROCm via HIP-NCCL (ensure your ROCm environment is set up correctly)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    print(f"Distributed Training Initialized: Rank={rank}, Local Rank={local_rank}, World Size={world_size}, Current Device: {torch.cuda.current_device()}")
    return device, rank, world_size

# --------------------------------
# Model Loading and FSDP Wrapping
# --------------------------------
model_path = "../llama3.2_vision_11b/"

print("Loading Model on CPU first...")
model = AutoModelForVision2Seq.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cpu")

# Initialize distributed training if running on multiple nodes/GPUs
if int(os.environ.get("WORLD_SIZE", 1)) > 1:
    device, rank, world_size = init_distributed()
else:
    device = torch.device("cuda:0")
    rank, world_size = 0, 1

processor = AutoProcessor.from_pretrained(model_path)
print(processor)
# --------------------------------
# Load CSV Data and Preprocess Reports
# --------------------------------
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')

def extract_findings_impression(report_text):
    findings_match = re.search(r'findings:\s*(.*?)(?=impression:|$)', report_text, re.DOTALL)
    impression_match = re.search(r'impression:\s*(.*)', report_text, re.DOTALL)
    findings_text = findings_match.group(1).strip() if findings_match and findings_match.group(1).strip() else "no findings provided."
    impression_text = impression_match.group(1).strip() if impression_match and impression_match.group(1).strip() else "no impression provided."
    return findings_text, impression_text

train_df[['findings', 'impression']] = train_df['report_text'].apply(lambda x: pd.Series(extract_findings_impression(x)))
val_df[['findings', 'impression']] = val_df['report_text'].apply(lambda x: pd.Series(extract_findings_impression(x)))

train_df = train_df[~((train_df['findings'] == "no findings provided.") & (train_df['impression'] == "no impression provided."))]
val_df = val_df[~((val_df['findings'] == "no findings provided.") & (val_df['impression'] == "no impression provided."))]

# --------------------------------
# Dataset Classes
# --------------------------------
class MimicCxrDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.actions = ["generate", "create", "produce", "provide", "compose"]
        self.details = ["a detailed", "a comprehensive", "a thorough", "an accurate", "a complete"]
        self.instructions = [
            "description of the findings in this X-ray image",
            "report on the impression from this X-ray",
            "radiology report including findings and impression",
            "medical analysis of the X-ray with findings and impression"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        findings_text = self.data.iloc[idx, 2]
        impression_text = self.data.iloc[idx, 3]
        image = Image.open(image_path).convert('RGB').resize((224, 224))
        findings_report = f"FINDINGS:\n{findings_text}\n"
        impression_report = f"IMPRESSION:\n{impression_text}\n"
        full_report = f"FINDINGS:\n{findings_text}\n\nIMPRESSION:\n{impression_text}\n"
        action = random.choice(self.actions)
        detail = random.choice(self.details)
        instruction = random.choice(self.instructions)
        prompt = f"Please {action} {detail} {instruction}."
        if "findings" in instruction:
            response = findings_report
        elif "impression" in instruction:
            response = impression_report
        else:
            response = full_report
        return {"image": image, "text": response, "prompt": prompt}


train1_dataset = MimicCxrDataset(train_df)
val1_dataset = MimicCxrDataset(val_df)

class FormattedMimicCxrDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        return format_data(sample)

system_message = "You are an expert radiographer. Describe accurately what you see in this image in detail."

def format_data(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": [
                {"type": "text", "text": sample["prompt"]},
                {"type": "image", "image": sample["image"]}
            ]},
            {"role": "assistant", "content": sample["text"]}
        ]
    }

train_dataset = FormattedMimicCxrDataset(train1_dataset)
val_dataset = FormattedMimicCxrDataset(val1_dataset)

# --------------------------------
# LoRA Fine-Tuning Preparation
# --------------------------------
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.01,
    r=8,
    bias="none",
    # Adjust target modules as needed for your model architecture.
    target_modules=["q_proj", "v_proj", "fc1", "fc2", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model; only the LoRA parameters will be trainable.
model = get_peft_model(model, peft_config)

# Patch model.forward to safely ignore bad kwargs
original_forward = model.forward
def patched_forward(*args, **kwargs):
    kwargs.pop("num_items_in_batch", None)
    return original_forward(*args, **kwargs)
model.forward = patched_forward

for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")


def collate_fn(examples):
    full_texts = []
    images = []
    image_token = "<|image|>"

    for ex in examples:
        system_msg = ex["messages"][0]["content"]
        user_content = ex["messages"][1]["content"]
        user_prompt = user_content[0]["text"]
        image = user_content[1]["image"]
        assistant_text = ex["messages"][2]["content"]

        prompt = f"{system_msg}\n{image_token}\n{user_prompt}"
        full_text = prompt + "\n" + assistant_text

        full_texts.append(full_text)
        images.append(image)

    tokenized = processor.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    labels = input_ids.clone()

    for i, ex in enumerate(examples):
        system_msg = ex["messages"][0]["content"]
        user_prompt = ex["messages"][1]["content"][0]["text"]
        prompt = f"{system_msg}\n{image_token}\n{user_prompt}"
        prompt_ids = processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        labels[i, :len(prompt_ids)] = -100

    # Use processor for images — it should return pixel_values + aspect_ratio_ids + aspect_ratio_mask
    image_inputs = processor(images, return_tensors="pt")

    device = input_ids.device

    pixel_values = image_inputs["pixel_values"].to(dtype=torch.bfloat16, device=device)
    aspect_ratio_ids = image_inputs.get("aspect_ratio_ids", torch.zeros(len(images), dtype=torch.long)).to(device)
    visual_length = processor.image_processor.size["height"] * processor.image_processor.size["width"] // (14 * 14)
    aspect_ratio_mask = image_inputs.get("aspect_ratio_mask", torch.ones((len(images), visual_length), dtype=torch.bool)).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "aspect_ratio_ids": aspect_ratio_ids,
        "aspect_ratio_mask": aspect_ratio_mask,
        "labels": labels,
    }



from transformers import TrainerCallback

# log epoch progress to help debugging
class EpochLoggerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"Starting epoch {int(state.epoch)} at step {state.global_step} on rank {rank}")

# --------------------------------
# Training Arguments
# --------------------------------
training_args = SFTConfig(
    output_dir = f"llama3.2-11b-instruct-finetune3000_new2-rank{rank}",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    warmup_steps=500,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="steps",                # Save every N steps instead of every epoch
    save_steps=2,                        # Save every 10 steps (you wrote 200 in comment)
    save_total_limit=2,                   # Only keep the last 2 checkpoints
    eval_strategy="no",                   # Disable evaluation temporarily
    learning_rate=1e-4,
    weight_decay=0.001,
    lr_scheduler_type="linear",
    bf16=True,
    tf32=False,
    max_grad_norm=0.5,
    max_seq_length=2048,
    save_on_each_node=True,
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    fsdp="full_shard",
    fsdp_config = {
    "fsdp_use_orig_params": False,
    "fsdp_backward_prefetch": "backward_pre",
    "fsdp_forward_prefetch": True,
    "fsdp_cpu_offload": True,
    "fsdp_state_dict_type": "sharded_state_dict",
    "fsdp_state_dict_config": {
        "offload_to_cpu": True,
        "rank0_only": False   
    }
}
)

training_args.remove_unused_columns = False
training_args.dataloader_pin_memory = True

# --------------------------------
# Trainer Initialization and Training
# --------------------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    processing_class=processor,
)

# Add epoch logger callback
trainer.add_callback(EpochLoggerCallback())

try:
    trainer.train()
except Exception as e:
    print(f"[Rank {rank}] ❌ Training failed: {e}")
    raise


if trainer.is_main_process:
    trainer.save_model("llama3.2-11b-instruct-finetune")