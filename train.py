"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, model_builder, model_trainer, utils
from torchvision import transforms
from pathlib import Path

if __name__ == "__main__":
    # setup hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 8
    HIDDEN_UNITS = 10
    LEARNING_RATE = 0.001

    # setup directories
    train_dir = Path("data/") / "cats_dogs" / "train"
    test_dir = Path("data/") / "cats_dogs" / "test"

    # setup tarfet device
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        num_workers=1,
    )

    # create model using model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
    ).to(device)

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

    # save the model using utils.py
    utils.save_model(model=model, target_dir="models", model_name="TinyVGG_model0.pth")
