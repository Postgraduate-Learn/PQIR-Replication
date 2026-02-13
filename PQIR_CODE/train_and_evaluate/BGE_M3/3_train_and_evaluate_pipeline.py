import os
import random
import joblib
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, precision_score
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers.modeling_outputs import SequenceClassifierOutput
from conf import conf


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors = 'pt', padding = 'max_length',
                                  truncation = True, max_length = self.max_length)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype = torch.float)
        }

    def __len__(self):
        return len(self.labels)


class WeightedBGEForSequenceClassification(torch.nn.Module):
    def __init__(self, model_name, num_labels, class_weights=None):
        super().__init__()
        self.num_labels = num_labels
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.backbone.config.hidden_size, num_labels)
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids, attention_mask = attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 使用[CLS] token
        logits = self.classifier(pooled_output)

        if labels is not None:
            # 确保权重设备一致
            class_weights = self.class_weights.to(labels.device)
            # 计算加权损失
            loss_fct = BCEWithLogitsLoss(pos_weight = class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))

            return SequenceClassifierOutput(
                loss = loss,
                logits = logits,
                hidden_states = outputs.hidden_states,
                attentions = outputs.attentions,
            )
        return SequenceClassifierOutput(
            logits = logits,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
        )


def init(seed):
    """设置随机种子的函数和环境"""
    # Python的随机数生成器
    random.seed(seed)
    # numpy的随机数生成器
    np.random.seed(seed)
    # PyTorch的随机数生成器 vbg
    torch.manual_seed(seed)
    # 当前GPU的随机数生成器
    torch.cuda.manual_seed(seed)
    # 所有GPU的随机数生成器
    torch.cuda.manual_seed_all(seed)
    # 以下设置可以确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # 设置环境
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def read_excel_data(train_data_paths, needed_columns):
    """读取excel数据"""
    data_frames = [pd.read_excel(file, usecols = needed_columns) for file in train_data_paths]
    data = pd.concat(data_frames, ignore_index = True)
    # 去除空值，并去除多余空格
    data = data.dropna(subset = ['Combined_Text'])
    data['Combined_Text'] = data['Combined_Text'].str.strip().astype(str)
    # 处理多标签
    data['labels'] = data[['Category_1', 'Category_2', 'Category_3']].values.tolist()
    data['labels'] = data['labels'].apply(lambda x: [label for label in x if pd.notna(label)])
    return data


def model_initialization(train_model_path, num_labels, y, device):
    # 计算类别权重
    label_counts = y.sum(axis = 0)
    total_samples = y.shape[0]
    pos_weight = (total_samples - label_counts) / label_counts
    pos_weight = torch.tensor(pos_weight, dtype = torch.float32).to(device)

    # 加载模型
    model = WeightedBGEForSequenceClassification(
        model_name = train_model_path,
        num_labels = num_labels,
        class_weights = pos_weight
    ).to(device)
    return model


def training_configuration(train_output_dir, train_logging_dir, seed):
    """训练参数配置"""
    return TrainingArguments(
        output_dir = train_output_dir,
        num_train_epochs = 15,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        warmup_ratio = 0.1,
        weight_decay = 0.001,
        learning_rate = 2e-5,
        lr_scheduler_type = "cosine",
        logging_dir = train_logging_dir,
        logging_steps = 10,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        load_best_model_at_end = True,
        metric_for_best_model = "mean_class_precision",
        greater_is_better = True,
        seed = seed,
        dataloader_num_workers = 0,
        save_total_limit = 1,
    )


def compute_metrics(eval_pred):
    """计算平均精确率"""
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    num_classes = labels.shape[1]
    class_precisions = []
    for i in range(num_classes):
        class_precision = precision_score(labels[:, i], predictions[:, i], zero_division = 0)
        class_precisions.append(class_precision)
    mean_precision = np.mean(class_precisions)
    return {"mean_class_precision": mean_precision}


def model_evaluation(trainer, val_dataset, y_val, mlb):
    """模型评估阶段"""
    predictions = trainer.predict(val_dataset)
    logits = predictions.predictions
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

    # 计算分类指标
    num_classes = y_val.shape[1]
    class_accuracies = []
    original_labels = mlb.classes_

    for i in range(num_classes):
        if y_val[:, i].sum() > 0:
            class_accuracy = accuracy_score(y_val[:, i], preds[:, i])
            class_accuracies.append(class_accuracy)
            print(f"类别 {original_labels[i]} 的准确率: {class_accuracy:.4f}")

    mean_accuracy = np.mean(class_accuracies) if class_accuracies else 0
    print(f"\n按类别计算的平均准确率: {mean_accuracy:.4f}")

    # 生成分类报告
    report = classification_report(y_val, preds, target_names = original_labels, zero_division = 0, output_dict = True)
    valid_metrics = [
        metrics for label, metrics in report.items()
        if label in original_labels and metrics['support'] > 0
    ]

    if valid_metrics:
        # 重新计算宏平均 (只基于有效类别)
        new_macro_p = np.mean([m['precision'] for m in valid_metrics])
        new_macro_r = np.mean([m['recall'] for m in valid_metrics])
        new_macro_f1 = np.mean([m['f1-score'] for m in valid_metrics])
        report['macro avg']['precision'] = new_macro_p
        report['macro avg']['recall'] = new_macro_r
        report['macro avg']['f1-score'] = new_macro_f1
    summary_rows = ['weighted avg', 'micro avg', 'samples avg', 'macro avg']
    valid_report = {label: metrics for label, metrics in report.items() if
                    (label in original_labels and metrics['support'] > 0) or label in summary_rows}
    report_df = pd.DataFrame(valid_report).transpose()
    print("\n分类报告：")
    print(report_df)

    return preds


def model_saving(model, tokenizer, mlb, model_save_path):
    """模型与数据保存阶段"""
    os.makedirs(model_save_path, exist_ok = True)
    model.backbone.save_pretrained(model_save_path)
    torch.save({
        'classifier_state_dict': model.classifier.state_dict(),
        'class_weights': model.class_weights,
        'num_labels': model.num_labels
    }, os.path.join(model_save_path, 'classifier.pt'))
    tokenizer.save_pretrained(model_save_path)
    joblib.dump(mlb, os.path.join(model_save_path, "label_encoder.pkl"))


def error_analysis(val_data, y_val, preds, mlb, error_data_path):
    """错误分类数据保存"""
    incorrect_indices = np.where((preds != y_val).any(axis = 1))[0]
    incorrect_data = {
        "Title": val_data.iloc[incorrect_indices]["Title"].tolist(),
        "Body": val_data.iloc[incorrect_indices]["Body"].tolist(),
        "Question_url": val_data.iloc[incorrect_indices]["Question_url"].tolist(),
        "Category_1": val_data.iloc[incorrect_indices]["Category_1"].tolist(),
        "Category_2": val_data.iloc[incorrect_indices]["Category_2"].tolist(),
        "Category_3": val_data.iloc[incorrect_indices]["Category_3"].tolist(),
        "Combined_Text": val_data.iloc[incorrect_indices]["Combined_Text"].tolist(),
        "True_Labels": [mlb.inverse_transform(np.array(y_val[i]).reshape(1, -1))[0] for i in incorrect_indices],
        "Predicted_Labels": [mlb.inverse_transform(np.array(preds[i]).reshape(1, -1))[0] for i in incorrect_indices],
    }
    directory = os.path.dirname(error_data_path)
    os.makedirs(directory, exist_ok = True)
    pd.DataFrame(incorrect_data).to_excel(error_data_path, index = False)
    print(f"\n错误分类数据已保存到: {error_data_path}")


def train_and_evaluate_pipeline():
    for seed in [42]:
        init(seed)
        print("BGE-M3:", seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用的设备: {device}")

        # --- 阶段1：数据预处理与分层划分 ---
        train_data_paths = conf.train_data_path
        needed_columns = ['Title', 'Body', 'Question_url', 'Category_1', 'Category_2', 'Category_3', 'Combined_Text']

        raw_train_data = read_excel_data(train_data_paths, needed_columns)
        X_full = np.array(raw_train_data['Combined_Text'].tolist())  # 转为 numpy 数组方便索引
        mlb = MultiLabelBinarizer()
        y_full = mlb.fit_transform(raw_train_data['labels'])
        num_labels = len(mlb.classes_)
        print("正在进行多标签分层切分 (Multilabel Stratified Split)...")
        msss = MultilabelStratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)

        for train_index, test_index in msss.split(X_full, y_full):
            X_train, X_test_internal = X_full[train_index], X_full[test_index]
            y_train, y_test_internal = y_full[train_index], y_full[test_index]

        print("\n" + "-" * 30)
        print("数据划分分布检查 (检查是否有类别缺失):")
        print(f"{'类别名称':<30} | {'训练集数量':<10} | {'测试集数量':<10} | {'测试集占比':<10}")
        print("-" * 70)

        train_counts = y_train.sum(axis = 0)
        test_counts = y_test_internal.sum(axis = 0)
        original_labels = mlb.classes_

        missing_in_test = []

        for i, label in enumerate(original_labels):
            total = train_counts[i] + test_counts[i]
            ratio = test_counts[i] / total if total > 0 else 0
            print(f"{str(label):<30} | {int(train_counts[i]):<10} | {int(test_counts[i]):<10} | {ratio:.1%}")

            if test_counts[i] == 0:
                missing_in_test.append(label)

        if missing_in_test:
            print(f"\n[警告] 以下类别在测试集中没有样本 (数量为0): {missing_in_test}")
        else:
            print("\n[成功] 所有类别在训练集和测试集中均有分布。")
        print("-" * 30 + "\n")

        tokenizer = AutoTokenizer.from_pretrained(conf.train_model_path)
        # 1.3 构建 Dataset
        train_dataset = TextDataset(X_train, y_train, tokenizer)  # 80% 用于训练
        internal_test_dataset = TextDataset(X_test_internal, y_test_internal, tokenizer)  # 20% 用于训练中验证/早停

        # 1.4 加载外部真实验证数据 (Real World Data)
        val_data_path = conf.test_data_paths
        val_data_real = read_excel_data(val_data_path, needed_columns)
        X_val_real = val_data_real['Combined_Text'].tolist()
        y_val_real = mlb.transform(val_data_real['labels'])
        real_val_dataset = TextDataset(X_val_real, y_val_real, tokenizer)

        # --- 阶段2：模型初始化 ---
        model = model_initialization(conf.train_model_path, num_labels, np.vstack([y_train]), device)

        # --- 阶段3：训练配置 ---
        training_args = training_configuration(conf.train_output_path, conf.train_logging_dir, seed)

        # --- 阶段4：创建Trainer并训练 ---
        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = internal_test_dataset,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)],
            compute_metrics = compute_metrics
        )
        trainer.train()

        # --- 阶段5：双重模型评估 ---

        # 结果 1: 内部测试集 (20% Split) 的效果
        print("\n" + "=" * 20 + " 结果1：内部测试集 (20% Split) 评估 " + "=" * 20)
        model_evaluation(trainer, internal_test_dataset, y_test_internal, mlb)

        # 结果 2: 真实验证集 (Real World Data) 的效果
        print("\n" + "=" * 20 + " 结果2：真实验证集 (Real Data) 评估 " + "=" * 20)
        # 这里接收 preds，用于后续生成错误报告（通常我们更关心真实数据的错误）
        preds_real = model_evaluation(trainer, real_val_dataset, y_val_real, mlb)

        # --- 阶段6：模型保存 ---
        model_saving(model, tokenizer, mlb, conf.model_save_path)
        #
        # # --- 阶段7：错误分析数据保存 (基于真实验证集) ---
        error_analysis(val_data_real, y_val_real, preds_real, mlb, conf.error_data_path)

        print("\n训练完成，模型和真实验证集的错误分类数据已保存")
        print("-----------------------------------------------------------------------")

if __name__ == "__main__":
    train_and_evaluate_pipeline()
