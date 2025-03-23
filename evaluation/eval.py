# python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/demo.json --port 8000 --max_new_tokens 4096 --batch_size 1024 --task api --use_chat_template --strict_prompt
# python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/demo.json --port 8000 --strict_prompt
# python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/demo.json --port 8000

import argparse
from tqdm import tqdm
import openai
from jinja2 import Template
import os
import json
from transformers import AutoTokenizer
from scorer import get_results

def postprocess_output(pred):
    pred = pred.strip().replace("</s>", "")
    return pred

def load_file(input_fp):
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    if isinstance(data, list):
        data = {'normal': data}
    for k, v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="unsloth/QwQ-32B")
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=8192)
    parser.add_argument('--max_tokens', type=int, default=0)
    parser.add_argument('--use_chat_template', action="store_true", default=True)
    parser.add_argument('--strict_prompt', action="store_true")
    parser.add_argument('--task', type=str, default='api')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--batch_size', type=int, default=256)    
    parser.add_argument('--url', type=str, default='http://127.0.0.1')
    parser.add_argument('--api_key', type=str, default='sk1234')
    
    args = parser.parse_args()
    
    print(f"Using local API server at port {args.port}")
    
    client = openai.OpenAI(
        base_url=f"{args.url}:{args.port}/v1", api_key=args.api_key
    )    
    
    if args.use_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
        template = Template(tokenizer.chat_template)
    
    def call_model(prompts, model, max_new_tokens=50, print_example=False):
        temperature = 0.6
        if print_example:
            print("Example:")
            print(prompts[0])
        
        if args.use_chat_template:
            prompts = [template.render(messages=[{"role": "system", "content": "detailed thinking on"},
                                               {"role": "user", "content": prom}],
                                      bos_token=tokenizer.bos_token, add_generation_prompt=True) for prom in prompts]
        
        if args.max_tokens > 0:
            new_prompts = []
            for prompt in prompts:
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                if len(input_ids) > args.max_tokens:
                    input_ids = input_ids[:args.max_tokens]
                    new_prompts.append(tokenizer.decode(input_ids))
                else:
                    new_prompts.append(prompt)
            prompts = new_prompts
        
        response = client.completions.create(
            model=args.model_name,
            prompt=prompts,
            temperature=temperature, max_tokens=max_new_tokens
        )
        
        preds = [x.text for x in response.choices]
        postprocessed_preds = [postprocess_output(pred) for pred in preds]
        return postprocessed_preds, preds
    
    input_data = load_file(args.eval_file)
    model = None
 
    final_results = []
    if args.strict_prompt:
        #query_prompt = "Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
        query_prompt='''
        You are absolutely correct. My apologies, that was an oversight. "Single best" strongly implies a single answer, even if multiple options *could* be argued as correct. However, to be *absolutely* certain we handle the (unlikely, but possible) case of a truly multiple-answer question correctly, we need to modify the prompt.

Here's the revised prompt, addressing the multiple-choice scenario explicitly:

```
You are a helpful and precise question-answering assistant. Your task is to answer the multiple-choice question provided below.

**Instructions:**

1.  **Carefully read the question and all provided options.**
2.  **Determine if the question requires a *single* answer or allows for *multiple* answers.**  Look for phrasing like "Select all that apply," "Choose the *one* best answer," "Which of the following are correct?", etc.  The question's wording is the primary guide.
3.  **Select the best answer(s) based on the question's instructions.**

**Output Format:**

Your response *must* adhere to the following formats, and contain *no other text*:

*   **Single Answer Questions:** If the question requires a single answer, your response *must* be a single line:

    ```
    The answer is **[letter]**
    ```

    Where `[letter]` is the letter (e.g., A, B, C, D) of the correct option.

**Important Considerations:**

*   If the question's wording is ambiguous about single vs. multiple answers, default to selecting only the *single best* answer.  Err on the side of choosing only one answer unless multiple answers are *clearly* indicated.
*   Do *not* provide any explanations, justifications, or additional commentary.
*   Do *not* attempt to rephrase the question or options.
*   Do *not* include any leading or trailing spaces.
*   Ignore any apparent formatting errors in the input.

**Question:**

{question}

**Options:**

{option_str}
'''

        query_prompt_old='''
        You are absolutely correct. My apologies, that was an oversight. "Single best" strongly implies a single answer, even if multiple options *could* be argued as correct. However, to be *absolutely* certain we handle the (unlikely, but possible) case of a truly multiple-answer question correctly, we need to modify the prompt.

Here's the revised prompt, addressing the multiple-choice scenario explicitly:

```
You are a helpful and precise question-answering assistant. Your task is to answer the multiple-choice question provided below.

**Instructions:**

1.  **Carefully read the question and all provided options.**
2.  **Determine if the question requires a *single* answer or allows for *multiple* answers.**  Look for phrasing like "Select all that apply," "Choose the *one* best answer," "Which of the following are correct?", etc.  The question's wording is the primary guide.
3.  **Select the best answer(s) based on the question's instructions.**

**Output Format:**

Your response *must* adhere to the following formats, and contain *no other text*:

*   **Single Answer Questions:** If the question requires a single answer, your response *must* be a single line:

    ```
    The answer is **[letter]**
    ```

    Where `[letter]` is the letter (e.g., A, B, C, D) of the correct option.

*   **Multiple Answer Questions:** If the question allows or requires multiple answers, your response *must* be a single line with the letters of all correct options, separated by commas, *without* spaces:

    ```
    The answer is **[letter1],[letter2],[letter3]**
    ```
    For example: `The answer is **A,C,D**`

* **Number options**: If options use numbers instead of letters. Use numbers:
   ```
   The answer is **[number]**
   ```
    For example, if options 1 and 3 are the correct answers: The answer is **1,3**

*   **No Correct Option:** If *no* option is correct, respond with:

    ```
    The answer is **none of the above**
    ```

**Important Considerations:**

*   If the question's wording is ambiguous about single vs. multiple answers, default to selecting only the *single best* answer.  Err on the side of choosing only one answer unless multiple answers are *clearly* indicated.
*   Do *not* provide any explanations, justifications, or additional commentary.
*   Do *not* attempt to rephrase the question or options.
*   Do *not* include any leading or trailing spaces.
*   Ignore any apparent formatting errors in the input.

**Question:**

{question}

**Options:**

{option_str}
'''

    else:
        query_prompt = "Please answer the following multiple-choice questions, ensuring your response concludes with the correct option in the format: 'The answer is A.'.\n{question}\n{option_str}"
        
    for idx in tqdm(range(len(input_data) // args.batch_size + 1)):
        batch = input_data[idx * args.batch_size:(idx + 1) * args.batch_size]
        if len(batch) == 0:
            break

        for item in batch:
            item['option_str'] = '\n'.join([f'{op}. {ans}' for op, ans in item['options'].items()])
            item["input_str"] = query_prompt.format_map(item)
        
        processed_batch = [item["input_str"] for item in batch]
    
        preds, _ = call_model(
            processed_batch, model=model, max_new_tokens=args.max_new_tokens, print_example=(idx == 0))
        
        for j, item in enumerate(batch):
            pred = preds[j]
            if len(pred) == 0:
                continue
            item["output"] = pred
            final_results.append(item)
    
    #task_name = os.path.split(args.model_name)[-1] + '_'
    #task_name += os.path.basename(args.eval_file).replace('.json', '') + f'_{args.task}' + ('_strict-prompt' if args.strict_prompt else '')
    task_name = os.path.basename(args.eval_file).replace('.json', '')
    save_path = f'output/{task_name}.json'
    
    with open(save_path, 'w') as fw:
        json.dump(final_results, fw, ensure_ascii=False, indent=2)
    
    get_results(save_path)

if __name__ == "__main__":
    main()
