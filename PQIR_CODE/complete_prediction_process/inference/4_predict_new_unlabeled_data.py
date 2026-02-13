import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModel
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
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length',
                                  truncation=True, max_length=self.max_length)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float)
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
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)

        if labels is not None:
            class_weights = self.class_weights.to(labels.device)
            loss_fct = BCEWithLogitsLoss(pos_weight=class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))

            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def setup_environment_and_load_resources(model_save_path):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    try:
        mlb = joblib.load(os.path.join(model_save_path, "label_encoder.pkl"))
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)

        classifier_info = torch.load(os.path.join(model_save_path, "classifier.pt"), map_location=device, weights_only=True)
        model = WeightedBGEForSequenceClassification(
            model_name=model_save_path,
            num_labels=classifier_info['num_labels'],
            class_weights=classifier_info['class_weights'].to(device)
        )
        model.classifier.load_state_dict(classifier_info['classifier_state_dict'])
        model = model.to(device)

        print(f"已加载模型: {model_save_path}")
        return device, model, tokenizer, mlb

    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)


def predict_new_data_and_save(input_file_path, output_file_path, model, tokenizer, mlb, device, max_length=256):
    df = pd.read_excel(input_file_path)
    df = df.dropna(subset=['Combined_Text'])
    df['Combined_Text'] = df['Combined_Text'].astype(str).str.strip()
    texts = df['Combined_Text'].tolist()

    dummy_labels = [[0] * len(mlb.classes_)] * len(texts)
    dataset = TextDataset(texts, dummy_labels, tokenizer, max_length)

    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(dataset, desc="Predicting New Data"):
            input_ids = batch['input_ids'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_logits.append(logits.cpu().numpy())

    all_logits = np.vstack(all_logits)
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    preds = (probs > 0.5).astype(int)

    predicted_labels = mlb.inverse_transform(preds)
    df['Predicted_Labels'] = ['; '.join(labels) for labels in predicted_labels]
    df.to_excel(output_file_path, index=False)
    print(f"\n预测结果已保存到: {output_file_path}")


def start_prediction_pipeline():
    model_save_path = conf.model_save_path
    input_file_path = conf.summarized_output_path
    output_file_path = conf.predicted_file_path
    print("开始进行模型结果预测...")
    device, model, tokenizer, mlb = setup_environment_and_load_resources(model_save_path)
    predict_new_data_and_save(input_file_path, output_file_path, model, tokenizer, mlb, device)
    print("模型结果预测完成！")


if __name__ == "__main__":
    start_prediction_pipeline()
