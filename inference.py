import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True, help='Your command for code generation!')
    args= parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser_opt()

    model_id =args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")

    instruction= args.instruction
    input=""

    prompt = f"""### Instruction:
    Use the Task below and the Input given to write the Response, which is a programming code that can solve the Task.

    ### Task:
    {instruction}

    ### Input:
    {input}

    ### Response:
    """

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=600, do_sample=True, top_p=0.9,temperature=0.3)

    print(f"Prompt:\n{prompt}\n")
    print(f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")
