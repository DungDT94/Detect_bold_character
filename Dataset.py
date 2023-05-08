import os
import requests
import zipfile
from pathlib import Path
import random
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import *
from torchinfo import summary
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader



def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        #print(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)


            test_pred_logits = model(X)


            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()


            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))


    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }


    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)


        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )


        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


    return results


if __name__ == "__main__":
    image_path = Path('/home/dungdinh/Documents/Prj2/data/data_train/')
    walk_through_dir('/home/dungdinh/Documents/Prj2/data/data_train')
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    data_transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    plot_transformed_images(image_path_list,
                        transform=data_transform,
                        n=3)

    train_data = datasets.ImageFolder(root=train_dir,  # target folder of images
                                      transform=data_transform,  # transforms to perform on data (images)
                                      target_transform=None)  # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=data_transform)

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
    print(train_data.classes)

    img, label = train_data[0][0], train_data[0][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")



    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=16,
                                  num_workers=1,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=16,
                                 num_workers=1,
                                 shuffle=False)


    img, label = next(iter(train_dataloader))


    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")


    model_0 = Model(input_shape=1,  # number of color channels (3 for RGB)
                      hidden_units=10,
                      output_shape=len(train_data.classes)).to(device)
    print(model_0)

    img_batch, label_batch = next(iter(train_dataloader))
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")

    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    model_0.eval()
    with torch.inference_mode():
        pred = model_0(img_single.to(device))

    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    NUM_EPOCHS = 100
    model_0 = Model(input_shape=1,  # number of color channels (3 for RGB)
                      hidden_units=10,
                      output_shape=len(train_data.classes)).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Train model_0
    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)

    torch.save(model_0.state_dict(), 'weights/model_weights_4.pth')
    #model_0_results.save('weights/model_weights.pth')
    # End the timer and print out how long it tookesults
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")






