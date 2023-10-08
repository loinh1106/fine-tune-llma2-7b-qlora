from datasets import load_dataset
from random import randrange

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM

from trl import SFTTrainer

from huggingface_hub import login
from dotenv import load_dotenv
import os

import argparse
from configs import configs as cfg

os.environ["HF_HUB_TOKEN"] = "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"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="iamtarun/python_code_instructions_18k_alpaca")
    parser.add_argument("--split", type=str, default="train[:90%]")
    parser.add_argument("--hf_repo", type=str, required=True)
     
    args= parser.parse_args()

    model_id = args.model_name 
    dataset_name = args.dataset
    dataset_split= args.split
    hf_model_repo= args.hf_repo
    device_map = {"": 0}
    
    
    load_dotenv()
    # Login to the Hugging Face Hub
    login(token=os.getenv("HF_HUB_TOKEN"))

    # Load dataset from the hub
    trainset = load_dataset(dataset_name, split=dataset_split)
    valset = load_dataset(dataset_name,split= 'train[:10%]')

    print(f"Training size: {len(trainset)}")
    print(f"Validation size: {len(valset)}")
    print(trainset[randrange(len(trainset))])
    print(valset[randrange(len(valset))])


    def format_instruction(sample):
        return f"""### Instruction:
    Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

    ### Task:
    {sample['instruction']}

    ### Input:
    {sample['input']}

    ### Response:
    {sample['output']}
    """

    print(format_instruction(trainset[randrange(len(trainset))]))
    print(format_instruction(valset[randrange(len(valset))]))

    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.use_4bit,
        bnb_4bit_use_double_quant=cfg.use_double_nested_quant,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache = False, device_map=device_map)
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    peft_config = LoraConfig(
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            r=cfg.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
        #save_steps=save_steps,
        logging_steps=cfg.logging_steps,
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        #tf32=True,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        group_by_length=cfg.group_by_length,
        lr_scheduler_type=cfg.lr_scheduler_type,
        disable_tqdm=cfg.disable_tqdm,
        report_to="tensorboard",
        seed=42
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=trainset,
        eval_dataset = valset,
        peft_config=peft_config,
        max_seq_length=cfg.max_seq_length,
        tokenizer=tokenizer,
        packing=cfg.packing,
        formatting_func=format_instruction,
        args=args,
    )
    print("Start Training")
    trainer.train()
    print("End Training")
    
    trainer.save_model()
    print("Model saved")

    del model
    del trainer
    import gc
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,    
    )

    merged_model = model.merge_and_unload()

    merged_model.save_pretrained("merged_model",safe_serialization=True)
    tokenizer.save_pretrained("merged_model")

    merged_model.push_to_hub(hf_model_repo)
    tokenizer.push_to_hub(hf_model_repo)