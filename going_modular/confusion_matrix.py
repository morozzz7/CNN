import torchmetrics
import seaborn as sns
import matplotlib.pyplot as plt
import torch

def create_confusion_matrix(model, dataloader, class_names):

    all_preds = []
    all_labels = []
    device = 'cpu'
    model.to(device)
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    num_classes = len(class_names)
    confmat = torchmetrics.ConfusionMatrix(task='multiclass', num_classes=num_classes)
    confusion_matrix = confmat(all_preds, all_labels)
    conf_matrix_np = confusion_matrix.cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_np, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.title('Матрица ошибок')
    plt.show()
