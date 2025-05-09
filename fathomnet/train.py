import os
import json
import random
import argparse
import pandas as pd
from PIL import Image

import torch
import pytorch_lightning as pl
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def create_subset_json(input_json_path, output_json_path, subset_size):
    with open(input_json_path, "r") as f:
        full_data = json.load(f)

    sampled_images = random.sample(full_data["images"], subset_size)
    sampled_image_ids = {img["id"] for img in sampled_images}

    sampled_annotations = [
        ann for ann in full_data["annotations"] if ann["image_id"] in sampled_image_ids
    ]

    subset_data = {
        "info": full_data.get("info", {}),
        "licenses": full_data.get("licenses", []),
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": full_data["categories"]
    }

    with open(output_json_path, "w") as f:
        json.dump(subset_data, f, indent=4)

class FathomNetDataset(Dataset):
    def __init__(self, csv_path, transform=None, label_encoder=None, is_test=False):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.is_test = is_test
        self.image_paths = self.data["path"].tolist()

        if not is_test:
            self.labels = self.data["label"].tolist()
            if label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.label_ids = self.label_encoder.fit_transform(self.labels)
            else:
                self.label_encoder = label_encoder
                self.label_ids = self.label_encoder.transform(self.labels)
        else:
            self.labels = None
            self.label_ids = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, self.image_paths[idx]
        else:
            label = self.label_ids[idx]
            return image, label

class FathomNetClassifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(100, num_classes)
        )
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main(args):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_annotations_df = pd.read_csv(args.train_csv)
    label_encoder = LabelEncoder().fit(train_annotations_df["label"].dropna())
    num_classes = len(label_encoder.classes_)

    full_dataset = FathomNetDataset(csv_path=args.train_csv, transform=train_transforms, label_encoder=label_encoder)
    targets = full_dataset.label_ids

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X=targets, y=targets))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = FathomNetClassifier(num_classes=num_classes)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_model")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.001)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model, train_loader, val_loader)

    test_dataset = FathomNetDataset(
        csv_path=args.test_csv,
        transform=val_transforms,
        label_encoder=label_encoder,
        is_test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model = FathomNetClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path, num_classes=num_classes
    ).to(device)

    predictions, ids = [], []
    best_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, image_ids = batch
            images = images.to(device)
            logits = best_model(images)
            preds = torch.argmax(logits, dim=1).tolist()
            predictions.extend(preds)
            ids.extend(image_ids)

    decoded_predictions = label_encoder.inverse_transform(predictions)

    submission_df = pd.read_csv(args.test_csv)
    submission_df["annotation_id"] = range(1, len(submission_df) + 1)
    submission_df["concept_name"] = decoded_predictions
    submission_df = submission_df.drop(["path", "label"], axis=1)
    submission_df.to_csv(args.output_csv, index=False)
    print(f"Saved submission to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FathomNet 2025 Training Pipeline")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training annotations CSV")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to the test annotations CSV")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Output CSV file for submission")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoaders")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    args = parser.parse_args()

    main(args)

