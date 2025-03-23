import json

def analyze_predictions(json_file):
    """
    分析 JSON 檔案中的模型預測結果。

    Args:
        json_file: 包含問題與預測答案的 JSON 檔案路徑。

    Returns:
        一個 tuple，包含 (總題數, 答對題數, 正確率)。
    """
    correct_count = 0

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_count = len(data)

    for item in data:
        answer_idx = item['answer_idx']
        output = item['output']
        if answer_idx == output:
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0

    return total_count, correct_count, accuracy


# 定義所有要處理的 JSON 檔案
file_paths = {
    "demo": "output/answer_demo.json"
}

# 初始化總計數
total_questions_all = 0
correct_answers_all = 0

# 分析每個檔案
results = {}

for dataset, file_path in file_paths.items():
    try:
        total_questions, correct_answers, accuracy = analyze_predictions(file_path)
        results[dataset] = (total_questions, correct_answers, accuracy)
        total_questions_all += total_questions
        correct_answers_all += correct_answers
    except FileNotFoundError:
        print(f"⚠️ 檔案未找到: {file_path}")
    except json.JSONDecodeError:
        print(f"⚠️ JSON 解析錯誤: {file_path}")

# 計算整體正確率
overall_accuracy = correct_answers_all / total_questions_all if total_questions_all > 0 else 0

# 輸出結果
print("\n📊 各資料集結果:")
for dataset, (total, correct, acc) in results.items():
    print(f"{dataset}: 總共有 {total} 道題目，模型答對了 {correct} 道題目，正確率為 {acc:.2%}")

print("\n📈 整體結果:")
print(f"總共有 {total_questions_all} 道題目")
print(f"模型總共答對了 {correct_answers_all} 道題目")
print(f"整體正確率為 {overall_accuracy:.2%}")
