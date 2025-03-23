# evaluation

## 0. install
- 安裝套件 (I) LOCAL
```bash=
mkdir -p $HOME/uv
cd $HOME/uv
export PATH=$PATH:$HOME/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv evaluation --python 3.11 && source $HOME/uv/evaluation/bin/activate && uv pip install --upgrade pip
uv pip install openai tqdm jinja2 transformers  python-dotenv  huggingface-hub
```
- 安裝套件 (II) HPC
```bash=
mkdir -p /work/$(whoami)/uv
cd /work/$(whoami)/uv
export PATH=$PATH:$HOME/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv evaluation --python 3.11 && source /work/$(whoami)/uv/evaluation/bin/activate && uv pip install --upgrade pip
uv pip install openai tqdm jinja2 transformers  python-dotenv  huggingface-hub
```

- **編輯 登錄HF KEY**
```bash
source $HOME/uv/evaluation/bin/activate
huggingface-cli login
```
- **下載套件**
```bash
#mkdir -p /home/$(whoami)/github
#cd /home/$(whoami)/github
mkdir -p /work/$(whoami)/github
cd /work/$(whoami)/github
git clone https://github.com/c00cjz00/hpc_evaluation
```

## 1. 啟動vllm server
```bash
#cd /home/$(whoami)/github/hpc_evaluation
cd /work/$(whoami)/github/hpc_evaluation
./vllm_server.sh
```

## 2. 確認模型是否運轉
```
# 輸出模型名稱
curl -X 'GET' "http://127.0.0.1:8000/v1/models" \
-H 'accept: application/json' -H "Authorization: Bearer sk1234" |jq

# 測試模型回應
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
-H "Authorization: Bearer sk1234" \
-H "Content-Type: application/json" \
-d '{ "model": "c00cjz00/phi-4-14b-it-offon-R1-m22k", "messages": [{"role": "user", "content": "123"}], "temperature": 0.6 }'
```

## 3. 分流檔案
```
jq '."MedMCQA_validation"[:3]' data/eval_data.json > data/demo.json
jq '."MedMCQA_validation"[:10000]' data/eval_data.json > data/MedMCQA_validation.json
jq '."MedQA_USLME_test"[:10000]' data/eval_data.json > data/MedQA_USLME_test.json
jq '."PubMedQA_test"[:10000]' data/eval_data.json > data/PubMedQA_test.json
jq '."MMLU-Pro_Medical_test"[:10000]' data/eval_data.json > data/MMLU-Pro_Medical_test.json
jq '."GPQA_Medical_test"[:10000]' data/eval_data.json > data/GPQA_Medical_test.json
```


## 4. 進行評估
```bash
#python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/demo.json --port 8000 --max_new_tokens 4096 --batch_size 1024
python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/demo.json --port 8000 --max_new_tokens 4096 --batch_size 1024 --strict_prompt
```

## 5. 答案校正
```
input="output/demo.json"
output="output/answer_demo.json"
jq -f evaluation/map.jq $input > $output
```

## 6. 統計分數 (demo)
```
python evaluation/answer_demo.py
```

## 7. 正式運轉測試
```
python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/MedMCQA_validation.json --port 8000 --strict_prompt
python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/MedQA_USLME_test.json --port 8000 --strict_prompt
python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/PubMedQA_test.json --port 8000 --strict_prompt
python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/MMLU-Pro_Medical_test.json --port 8000 --strict_prompt
python evaluation/eval.py --url http://127.0.0.1 --api_key sk1234 --model_name "c00cjz00/phi-4-14b-it-offon-R1-m22k" --eval_file data/GPQA_Medical_test.json --port 8000 --strict_prompt
```

## 8. 答案校正
```
input="output/MedMCQA_validation.json"
output="output/answer_MedMCQA_validation.json"
jq -f evaluation/map.jq $input > $output

input="output/MedQA_USLME_test.json"
output="output/answer_MedQA_USLME_test.json"
jq -f evaluation/map.jq $input > $output

input="output/PubMedQA_test.json"
output="output/answer_PubMedQA_test.json"
jq -f evaluation/map.jq $input > $output

input="output/MMLU-Pro_Medical_test.json"
output="output/answer_MMLU-Pro_Medical_test.json"
jq -f evaluation/map.jq $input > $output

input="output/GPQA_Medical_test.json"
output="output/answer_GPQA_Medical_test.json"
jq -f evaluation/map.jq $input > $output
```

## 9. 統計分數
```
python evaluation/answer.py
```


## 10. 關閉vllm
```
ps aux | grep /opt/venv/bin/vllm | awk '{print $2}' | xargs kill -9
```
