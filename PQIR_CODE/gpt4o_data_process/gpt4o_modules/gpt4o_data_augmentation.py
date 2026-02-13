import pandas as pd
from openai import OpenAI
from typing import List, Optional
from conf import conf
import time


def load_and_preprocess_data(file_paths: List[str]) -> pd.DataFrame:
    """读取多个Excel文件并合并为一个DataFrame，预处理类别列"""
    # 读取并合并所有文件
    dataframes = [pd.read_excel(file) for file in file_paths]
    all_data = pd.concat(dataframes, ignore_index=True)

    # 处理类别列：去除空格并转为字符串
    for col in conf.sample_category_columns:
        all_data[col] = all_data[col].astype(str).str.strip()

    # 合并标题和正文作为完整文本
    all_data['Combined_Text'] = 'Title:' + all_data['Title'] + '\n' + 'Body Text:' + all_data['Body']
    return all_data


def identify_rare_categories(data: pd.DataFrame) -> List[str]:
    """识别样本数量少于阈值的少样本类别"""
    # 合并所有类别列的非空值，并清洗
    all_categories = pd.concat([
        data[col].dropna().astype(str).str.strip() for col in conf.sample_category_columns
    ])

    # 去除字符串形式的 'nan' 和空字符串
    all_categories = all_categories[~all_categories.isin(["", "nan", "NaN", "None"])]

    # 统计每个类别的样本数量
    category_counts = all_categories.value_counts()
    print(f"类别分布统计：\n{category_counts}")

    # 找出样本数少于阈值的类别
    rare_categories = category_counts[category_counts < conf.min_sample_threshold].index.tolist()
    print(f"少样本类别({len(rare_categories)}个): {rare_categories}")
    return rare_categories




def generate_augmented_text(original_text: str, client: OpenAI) -> Optional[str]:
    """使用OpenAI API生成与原文语义相似的增强文本"""
    prompt = f"""Generate a new Stack Overflow question based on the following technical question, but with a different wording while maintaining the same meaning.
       Requirements:
       1. The generated content should be consistent with the original meaning or similar to the original text but rephrased differently.
       2. Maintain a professional tone and adhere to the format of Stack Overflow questions.
       3. Ensure the language is natural and fluent.
       4. Ensure semantic similarity is above 95%
        Original Question: {original_text}
       """

    try:
        response = client.chat.completions.create(
            model=conf.gpt_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT生成失败: {e}")
        return None


def augment_data(data: pd.DataFrame, rare_categories: List[str], client: OpenAI) -> pd.DataFrame:
    """为包含少样本类别的数据生成增强版本"""
    augmented_rows = []

    # 遍历每一条数据
    for _, row in data.iterrows():
        # 获取该样本涉及的所有少样本类别
        sample_rare_categories = [
            row[col] for col in conf.sample_category_columns
            if pd.notna(row[col]) and row[col] in rare_categories
        ]

        # 如果样本涉及少样本类别，则生成增强数据
        if sample_rare_categories:
            for _ in range(conf.augmentation_sample_count):
                # 生成增强文本
                new_text = generate_augmented_text(row['Combined_Text'], client)
                if new_text:
                    # 创建新的数据行，仅替换文本内容
                    new_row = row.copy()
                    new_row['Combined_Text'] = new_text
                    augmented_rows.append(new_row)

    # 合并原始数据和增强数据
    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        print(f"成功生成{len(augmented_rows)}条增强数据")
        return pd.concat([data, augmented_df], ignore_index=True)
    else:
        print("未生成任何增强数据")
        return data


def save_data(data: pd.DataFrame, output_path: str) -> None:
    # 确保类别列的空值保存为空字符串
    data = data.replace('nan', '')
    # 保存到Excel
    data.to_excel(output_path, index=True)
    print(f"增强数据已保存至 {output_path}")


def pipeline_augmentation():
    # 初始化OpenAI客户端
    client: OpenAI = OpenAI(
        api_key=conf.api_key,
        base_url=conf.api_base_url
    )
    start_time = time.perf_counter()
    print("开始执行数据增强流程...")

    # 读取和预处理数据
    print("正在读取和预处理数据...")
    all_data = load_and_preprocess_data(conf.augmentation_input_paths)

    max_iterations = 100  # 设置最大迭代轮数，防止费用爆炸或无限循环
    current_iter = 0

    while True:
        current_iter += 1
        print(f"\n>>> 开始第 {current_iter} 轮 样本检测与增强 <<<")

        # 1. 基于“当前所有数据”重新识别少样本类别
        # (每次循环都会重新统计，包括上一轮生成的增强数据)
        rare_categories = identify_rare_categories(all_data)

        # 2. 如果没有少样本类别了，或者达到最大轮数，则停止
        if not rare_categories:
            print("所有类别均已满足阈值要求，增强结束。")
            break

        if current_iter > max_iterations:
            print(f"已达到最大迭代次数 ({max_iterations})，强制结束增强以控制成本。")
            break

        # 3. 记录当前数据量，用于后续判断
        prev_data_count = len(all_data)

        # 4. 执行增强
        all_data = augment_data(all_data, rare_categories, client)

        # 5. 安全检查：如果本轮没有新增任何数据（可能是API错误或无法生成），防止死循环
        if len(all_data) == prev_data_count:
            print("警告：本轮未生成任何新数据，停止循环。")
            break

    # 保存结果
    print("正在保存最终增强数据...")
    save_data(all_data, conf.augmented_output_path)

    print("数据增强流程已完成!")
    end_time = time.perf_counter()
    print('Running time:\n', end_time - start_time)

# 主程序执行流程
if __name__ == "__main__":
    pipeline_augmentation()
