from seisml.datasets import EarthquakeDataset
import torch
from tqdm import trange, tqdm
import warnings
import pprint
import os
from torch.utils.data import DataLoader
from seisml.utility.parser import construct_parser
import json
from seisml.utility import utils
from argparse import Namespace
import pickle
import numpy as np
import matplotlib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from pycm import ConfusionMatrix

matplotlib.use('agg')

import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

pp = pprint.PrettyPrinter(indent=4)
torch.manual_seed(0)
warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.monitor_interval = 0

parser = construct_parser()
args = parser.parse_args()

with open(os.path.join(args.output_directory, 'args.json'), 'r') as f:
    args = json.load(f)

args = Namespace(**args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = EarthquakeDataset(folder=args.dataset_directory,
                            transforms=args.transforms,
                            augmentations='',
                            length=args.length,
                            mode='test',
                            split=args.split)

model, _, _ = utils.load_model(args.output_directory, device_target='cuda')
model.to(device)
model.eval()


def get_embeddings(dataset, model):
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=args.num_workers,
                            drop_last=False)

    epoch_loss = []
    embeddings = []
    labels = []
    for data_dict in dataloader:
        data = data_dict['data'].to(device).requires_grad_().float()
        label = data_dict['label'].to(device).float()
        output = model(data).squeeze(1)
        embeddings.append(output.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    return np.vstack(embeddings), np.vstack(labels)


def train_svm(embeddings, labels):
    print(embeddings.shape, labels.shape)
    # svc = SVC(kernel='rbf')
    svc = KNeighborsClassifier(n_neighbors=11)
    svc.fit(embeddings, np.argmax(labels, axis=-1))
    return svc


print(args.split)
print('Getting train embeddings')
embeddings, labels = get_embeddings(dataset, model)
plt.clf()
pca = utils.visualize_embedding(embeddings, labels, os.path.join(args.output_directory, 'tr_viz.png'))

print('Training KNN')
svc = train_svm(embeddings, labels)

print('Predicting on train embeddings with KNN')
predictions = svc.predict(embeddings)
ground_truth = np.argmax(labels, axis=-1)

cm = ConfusionMatrix(predict_vector=predictions, actual_vector=ground_truth)
print(str(cm))

print('Getting test embeddings')
dataset.toggle_split()
embeddings, labels = get_embeddings(dataset, model)
plt.clf()
utils.visualize_embedding(embeddings, labels, os.path.join(args.output_directory, 'tt_viz.png'), pca=pca)

print('Predicting on test embeddings with KNN')
predictions = svc.predict(embeddings)
ground_truth = np.argmax(labels, axis=-1)

try:
    cm = ConfusionMatrix(predict_vector=predictions, actual_vector=ground_truth)
    results = str(cm)
except:
    results = str(sum(predictions == ground_truth) / len(ground_truth))

with open(os.path.join(args.output_directory, 'results.txt'), 'w') as f:
    f.write(results)

print("Visualizing with KDE plot")

font = {'family': 'normal',
        'weight': 'normal',
        'size': 14}

matplotlib.rc('font', **font)

dataset.toggle_split()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
##ax.set_facecolor('grey')

embeddings, labels = get_embeddings(dataset, model)
pca = PCA(n_components=2)
pca.fit(embeddings)
output = pca.transform(embeddings[labels.argmax(axis=-1) == 0])
sns.kdeplot(output[:, 0], output[:, 1], cmap="Blues_r", shade=False, linestyles="-", linewidths=1, shade_lowest=False,
            cut=3, gridsize=100)
# sns.kdeplot(output[:, 0], output[:, 1], cmap="gray", shade=False, linestyles=":", linewidths=1.5, shade_lowest=False, cut=3, gridsize=100)

label_patches = []

label_patch = mpatches.Patch(
    color=sns.color_palette('Blues_r')[2],
    #        linestyle='--',
    #        marker='-',
    label='Negative (train)')
label_patches.append(label_patch)

output = pca.transform(embeddings[labels.argmax(axis=-1) == 1])
sns.kdeplot(output[:, 0], output[:, 1], cmap="Reds_r", shade=False, linestyles="-", linewidths=1, shade_lowest=False,
            cut=3, gridsize=100)
# sns.kdeplot(output[:, 0], output[:, 1], cmap="gray", shade=False, linestyles="--", linewidths=1.5, shade_lowest=False, cut=3, gridsize=100)

label_patch = mpatches.Patch(
    color=sns.color_palette('Reds_r')[2],
    label='Positive (train)')
label_patches.append(label_patch)

dataset.toggle_split()
print(len(dataset))

embeddings, labels = get_embeddings(dataset, model)
output = pca.transform(embeddings[labels.argmax(axis=-1) == 0])
# plt.scatter(output[:, 0], output[:, 1], color='black', marker='o', facecolors='none', edgecolors='black', linewidth=1.5, s=100)
plt.scatter(output[:, 0], output[:, 1], color='blue', marker='o', facecolors='none', linewidth=1.5, edgecolors='blue',
            s=100)

# label_patches.append(mlines.Line2D([], [], color='black', marker='o',markerfacecolor='none', linestyle='None',
#                          markersize=10, label='Negative (test)'))
label_patches.append(
    mlines.Line2D([], [], color='blue', marker='o', linewidth=1.5, markerfacecolor='none', linestyle='None',
                  markersize=10, label='Negative (test)'))

output = pca.transform(embeddings[labels.argmax(axis=-1) == 1])
# plt.scatter(output[:, 0], output[:, 1], color='white', marker='o', facecolors='none', edgecolors='white', linewidth=1.5, s=100)
plt.scatter(output[:, 0], output[:, 1], color='red', marker='x', linewidth=1.5, edgecolors='red', s=100)

# label_patches.append(mlines.Line2D([], [], color='white', marker='o', markerfacecolor='none', markeredgecolor='white', linestyle='None',
#                          markersize=10, label='Positive (test)'))
label_patches.append(mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                                   markersize=10, linewidth=1.5, label='Positive (test)'))

plt.title('PCA of embeddings for train and test')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')

legend = plt.legend(handles=label_patches, numpoints=1)
frame = legend.get_frame()
# frame.set_facecolor('lightgray')
leg_lines = legend.get_lines()
plt.setp(leg_lines, linewidth=1.5)

plt.xlim([-1.0, 2.2])
plt.ylim([-.85, .85])

plt.tight_layout()
plt.savefig(os.path.join(args.output_directory, 'combined.pdf'))
