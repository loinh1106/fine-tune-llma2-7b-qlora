use_4bit = True
    # Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Activate nested quantization for 4-bit base models (double quantization)
use_double_nested_quant = False

################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 64
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# TrainingArguments parameters
################################################################################
# Output directory where the model predictions and checkpoints will be stored
output_dir = 'outputs'
# Number of training epochs
num_train_epochs = 10 #1
# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = True
# Batch size per GPU for training
per_device_train_batch_size = 4
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4 # 2
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4 ##1e-5
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule
lr_scheduler_type = "cosine" #"constant"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = False #True
# Save checkpoint every X updates steps
save_steps = 0
# Log every X updates steps
logging_steps = 200
# Disable tqdm progress bars, 
disable_tqdm=False
################################################################################
# SFTTrainerparameters
################################################################################
# Maximum sequence length to use
# max sequence length for model and packing of the dataset
max_seq_length = 512 # None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = True #False
