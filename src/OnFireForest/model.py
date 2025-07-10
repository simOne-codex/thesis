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
from dataset import PiedmontDataset

# CNN Feature Extractor
class ReduceToOnePixelNet(nn.Module):
    def __init__(self, in_channels=36, img_size=256):
        super(ReduceToOnePixelNet, self).__init__()

        layers = []

        channels = in_channels
        while(img_size > 1):  # 256 -> 128 -> ... -> 1
            # out_channels = base_channels * min(2**i, 8) 
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            img_size = int(img_size/2)

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        r = self.encoder(x)
        return r.view(1, -1)


# Image + Tabular Dataset
class TabularImageDataset(Dataset):
    pass
#     def __init__(self, dataframe, images, transform=None):
#         self.df = dataframe.reset_index(drop=True)
#         self.images = images
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((32, 32)),
#             transforms.ToTensor()
#         ])

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         tabular_data = self.df.drop(columns='target').iloc[idx].values.astype(np.float32)
#         label = self.df['target'].iloc[idx]
#         image = self.images[idx]

#         if isinstance(image, np.ndarray):
#             image = Image.fromarray(image.astype('uint8'))
#         image = self.transform(image)

#         return tabular_data, image, label



# Hybrid Model Class
class TabularImageModel(BaseEstimator, ClassifierMixin):
    def __init__(self, image_feature_dim=256, rf_params=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_feature_dim = image_feature_dim
        self.rf_params = rf_params or {'n_estimators': 100} # input the parameters selected in the rf only part
        self.cnn = ReduceToOnePixelNet().to(self.device)
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

        """ Here insert the final model selected in the rf only part
        self.rf = RandomForestClassifier(**self.rf_params)
        self.rf.fit(X_combined, y)
        return self
        """
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