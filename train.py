"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, model_builder, model_trainer, utils
from torchvision import transforms
from pathlib import Path

# setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 128
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = 1

# setup directories
train_dir = Path("data/") / "cats_dogs" / "train"
test_dir = Path("data/") / "cats_dogs" / "test"

# setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" The target  device is {device}")

# creat transforms
data_trasnform = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor()]
)

# create DataLoaders using data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_trasnform,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# create model using model_builder.py
model = model_builder.TinyVGG(
    input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
).to(device)
print(model)


# try  a single forward pass on a single image to test the model
img_batch, label_batch = next(iter(train_dataloader))
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}\n")
model.eval()
with torch.inference_mode():
    pred = model(img_single.to(device))

print(f"Output logits :\n{pred}\n")
print(f"Output predictoin probabilites:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim= 1),dim=1)}\n")
print(f"Actual label:\n{label_single}")

# use torchinfo to get an idea of the shapes going through our model
from torchinfo import summary

summary(model, input_size=[1, 3, 64, 64])

# set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# start training using model model_trainer.py
model_trainer.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device,
)
print(f" Traing is done")
# save the model using utils.py
utils.save_model(model=model, target_dir="models", model_name="TinyVGG_model0.pth")
