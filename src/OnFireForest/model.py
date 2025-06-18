from PointDataLoader import PointDataLoader
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# CNN Feature Extractor
class SimpleCNN(nn.Module):
    def __init__(self, input_dim = 36, output_dim=128):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Output: 32x1x1
        )
        self.fc = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


"""
# Image + Tabular Dataset
class TabularImageDataset(Dataset):
    def __init__(self, dataframe, images, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.images = images
        self.transform = transform or transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tabular_data = self.df.drop(columns='target').iloc[idx].values.astype(np.float32)
        label = self.df['target'].iloc[idx]
        image = self.images[idx]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'))
        image = self.transform(image)

        return tabular_data, image, label
"""


# Hybrid Model Class
class TabularImageModel(BaseEstimator, ClassifierMixin):
    def __init__(self, image_feature_dim=128, rf_params=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_feature_dim = image_feature_dim
        self.rf_params = rf_params or {'n_estimators': 100}
        self.cnn = SimpleCNN(output_dim=image_feature_dim).to(self.device)
        self.rf = None  # Will initialize after training

    def fit(self, df, images):
        dataset = TabularImageDataset(df, images)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        cnn_features = []
        tabular_features = []
        labels = []

        self.cnn.eval()
        with torch.no_grad():
            for tabular_data, image_data, label in loader:
                image_data = image_data.to(self.device)
                encoded = self.cnn(image_data).cpu().numpy()
                cnn_features.append(encoded)
                tabular_features.append(tabular_data.numpy())
                labels.append(label.numpy())

        X_image = np.vstack(cnn_features)
        X_tabular = np.vstack(tabular_features)
        y = np.hstack(labels)

        X_combined = np.hstack([X_tabular, X_image])

        self.rf = RandomForestClassifier(**self.rf_params)
        self.rf.fit(X_combined, y)
        return self

    def predict(self, df, images):
        dataset = TabularImageDataset(df, images)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        cnn_features = []
        tabular_features = []

        self.cnn.eval()
        with torch.no_grad():
            for tabular_data, image_data, _ in loader:
                image_data = image_data.to(self.device)
                encoded = self.cnn(image_data).cpu().numpy()
                cnn_features.append(encoded)
                tabular_features.append(tabular_data.numpy())

        X_image = np.vstack(cnn_features)
        X_tabular = np.vstack(tabular_features)
        X_combined = np.hstack([X_tabular, X_image])

        return self.rf.predict(X_combined)