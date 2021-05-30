import argparse
import os
from datasets import OralCancerImageDataset
from models import ConvNeuralNetwork
from optimizers import train, validate
from transforms import DiscreteRotationTransform

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skl_m
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--validate_epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_cpu', type=int, default=0)

opt = parser.parse_args()

print(opt)

model_path = os.path.join(os.getcwd(), f'models/{opt.model}.pth')
data_path = os.path.join(os.getcwd(), 'data')
train_dir = os.path.join(data_path, 'train')
labels_file = os.path.join(data_path, 'train.csv')

train_transform = transforms.Compose([
    DiscreteRotationTransform(angles=[-90, 90, 180]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

train_dataset = OralCancerImageDataset(img_dir=train_dir,
                                       labels_file=labels_file)

train_data_length = int(len(train_dataset) * 0.75)
validation_data_length = len(train_dataset) - train_data_length

train_data, validation_data = random_split(
    train_dataset, [train_data_length, validation_data_length])

train_dataloader = DataLoader(train_data, batch_size=opt.batch_size,
                              shuffle=True, num_workers=opt.n_cpu)
validation_dataloader = DataLoader(validation_data, batch_size=opt.batch_size,
                                   shuffle=True, num_workers=opt.n_cpu)

model = ConvNeuralNetwork().to('cuda')
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
loss_fn = nn.CrossEntropyLoss()
epochs = opt.epochs

training_losses = np.zeros(epochs)
training_metrics = np.zeros(epochs)
validation_losses = []
validation_metrics = []

for epoch in range(epochs):
    print(f'Running epoch {epoch+1}')
    training_metrics[epoch], training_losses[epoch] =\
        train(train_dataloader, model, loss_fn,
              optimizer, 'cuda', verbose=False)

    if epoch % opt.validate_epochs == 0:
        validation_metric, validation_loss =\
            validate(validation_dataloader, model, loss_fn,
                     'cuda', verbose=False)
        validation_metrics.append(validation_metric)
        validation_losses.append(validation_loss)

    torch.save(model.state_dict(), model_path)

print('\nTraining complete.\n')

if not os.path.isdir(f'output/{opt.model}'):
    os.makedirs(f'output/{opt.model}')

plt.figure()
plt.plot(np.linspace(1, epochs, len(training_losses)),
         training_losses,
         label='Training')
plt.plot(np.linspace(1, epochs, len(validation_losses)),
         validation_losses,
         label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'output/{opt.model}/model_losses.png')

plt.figure()
plt.plot(np.linspace(1, epochs, len(training_metrics)),
         training_metrics,
         label='Training')
plt.plot(np.linspace(1, epochs, len(validation_metrics)),
         validation_metrics,
         label='Validation')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.savefig(f'output/{opt.model}/model_metrics.png')

model.eval()
all_y_true = []
all_y_pred = []
all_y_pred_binary = []

for batch in validation_dataloader:
    X, y = batch['image'], batch['label']
    X, y = X.to('cuda'), y.to('cuda')
    all_y_true += y.tolist()

    prediction = model(X)
    y_pred = nn.Softmax(dim=1)(prediction)
    all_y_pred += y_pred[:, 1].tolist()
    all_y_pred_binary += y_pred.argmax(1).tolist()

auc = skl_m.roc_auc_score(all_y_true, all_y_pred)
accuracy = skl_m.accuracy_score(all_y_true, all_y_pred_binary)
precision = skl_m.precision_score(all_y_true, all_y_pred_binary)
recall = skl_m.recall_score(all_y_true, all_y_pred_binary)
f1 = skl_m.f1_score(all_y_true, all_y_pred_binary)

print('Final validation metrics')
print(f' AUC: {auc:.4f}')
print(f' Accuracy: {accuracy:.4f}')
print(f' Precision: {precision:.4f}')
print(f' Recall: {recall:.4f}')
print(f' F1: {f1:.4f}')
