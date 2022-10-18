import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import Model
import argparse
import sys
import cv2
if(torch.cuda.is_available()):
  device=torch.device('cuda:0')
  print("GPU")
else:
  device=torch.device('cpu')
  print('CPU')
# data -> import the cifar dataset and prepare then for an acceptable input to the model.
# model architecture -> decide on number of layers and kind
# training -> decide on optimizer, loss function, regularization, learning rate.
# testing, prediction -> run model and print results in thr required format
# Possible changes -> num of layers, dimensions of layers, normalisation, dropout, activation, softmax, data transform,
# data augmentation, learning rate, bath size, weight decay, epochs

batch_size = 20
learning_rate = 0.001
epochs = 15
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
model_path = './model/cifar_net.pth'

parser = argparse.ArgumentParser()
parser.add_argument('command', choices=["train", "test"])
parser.add_argument('test_image', nargs="?")
args = parser.parse_args()


def prepare_dataset():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # Check if you can add more transforms to augment data

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


model = Model(3072).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


def train(train_data):
    loss = 0
    acc = 0
    for i, data in enumerate(train_data, 0):
        # sys.stdout.write('\r')
        # sys.stdout.write("Training: [%-50s] %d%%" % ('=' * int(50 * (i+1) / len(train_data)),
        #                                              int(100*(i + 1) / len(train_data))))
        # sys.stdout.flush()
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss_ = loss_function(output, labels)
        loss_.backward()
        optimizer.step()
        loss += loss_.item()
        y = []
        labels_copy = labels.cpu().numpy()
        for x in output:
            y.append(int((x == torch.max(x)).nonzero()[0][0]))
        y = np.asarray(y)
        y_temp = [y[i]==labels_copy[i] for i in range(batch_size)]
        acc += np.sum(y_temp)
    scheduler.step()
    return acc, loss


def test(test_data):
    loss = 0
    acc = 0
    with torch.no_grad():
        for i, data in enumerate(test_data):
            # sys.stdout.write('\r')
            # sys.stdout.write("Testing: [%-50s] %d%%" % ('=' * int(50 * (i+1) / len(test_data)),
            #                                              int(100 * (i + 1) / len(test_data))))
            # sys.stdout.flush()
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss_ = loss_function(output, labels)
            loss += loss_.item()
            y = []
            labels_copy = labels.cpu().numpy()
            for x in output:
                y.append(int((x == torch.max(x)).nonzero()[0][0]))
            y = np.asarray(y)
            y_temp = [y[i]==labels_copy[i] for i in range(batch_size)]
            acc += np.sum(y_temp)
    return acc, loss


def predict(inputs):
    model2 = Model(3072)
    model2.load_state_dict(torch.load(model_path))
    image_ = cv2.resize(inputs, (32, 32))
    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    inputs = transform(image_)
    inputs = inputs.unsqueeze(0)
    model2.eval()
    output = model2(inputs)
    predicted = int((output[0] == torch.max(output[0])).nonzero()[0][0])
    print('Predicted: ' f'{classes[predicted]:5s}')


if __name__ == '__main__':
    mode = args.command
    if mode == "train":
        train_, test_ = prepare_dataset()
        for epoch in range(epochs):
            train_acc, train_loss = train(train_)
            test_acc, test_loss = test(test_)
            sys.stdout.write('\r')
            print(f'\n{epoch + 1}/{epochs}'
                  f', Train Loss: {train_loss / (batch_size*len(train_)):.3f}'
                  f', Train Acc%: {100 * train_acc / (batch_size*len(train_)):.3f}'
                  f', Test Loss: {test_loss / (batch_size*len(test_)):.3f}'                  
                  f', Test Acc%: {100 * test_acc / (batch_size*len(test_)):.3f}')
            torch.save(model.state_dict(), model_path)
    elif mode == "test":
        image = args.test_image
        image = cv2.imread(image)
        if image is None:
            print("No image found")
        else:
            predict(image)
