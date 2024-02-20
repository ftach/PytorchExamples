""" Stores the functions used to train the U-Net model. """

# import the necessary packages
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

from monai.optimizers import ExponentialLR
from monai.losses import DiceLoss

from dataset import *
from unetr import *
from vnet import *
import config
from metrics import *

print(config.DEVICE)

train_ids, test_ids, val_ids = train_test_val_split(data_ids)

training_generator = CustomDataset(train_ids)
valid_generator = CustomDataset(val_ids, training=False)
test_generator = CustomDataset(test_ids, training=False)

print("Dataset loaded.")

# index = 10
# n_slice = 75
# n_slice = random.randint(0, config.IMG_SIZE-1)
# X_train = training_generator.__getitem__(index)[0][0]
# y_train = training_generator.__getitem__(index)[1][0]
# flatten_y_train = np.argmax(y_train, axis=-1)
# print(flatten_y_train.shape)
#
# plt.figure(figsize=(12, 8))
# plt.subplot(151)
# plt.imshow(X_train[n_slice, :, :, 0], cmap='bone')
# plt.title('Image FLAIR')
# plt.subplot(152)
# plt.imshow(X_train[n_slice, :, :, 1], cmap='bone')
# plt.title('Image T1CE')
# plt.subplot(153)
# plt.imshow(X_train[n_slice, :, :, 2], cmap='bone')
# plt.title('Image T1')
# plt.subplot(154)
# plt.imshow(X_train[n_slice, :, :, 3], cmap='bone')
# plt.title('Image T2')
# plt.subplot(155)
# plt.imshow(flatten_y_train[n_slice, :, :])
# plt.title('Mask')
# plt.show()


# initialize our UNetR model
# model = UNETR(in_channels=config.MODALITIES,
#              out_channels=config.NUM_CLASSES,
#              img_size=(config.IMG_SIZE, config.IMG_SIZE, config.IMG_SIZE),
#              conv_block=True,
#              dropout_rate=0.1,).to(config.DEVICE)

# initialize our Vnet model
model = VNet(in_channels=config.MODALITIES,
             out_channels=config.NUM_CLASSES).to(config.DEVICE)

# initialize loss function and optimizer
lossFunc = DiceLoss()
opt = AdamW(model.parameters(), lr=config.INIT_LR)
scheduler = ExponentialLR(opt, end_lr=1e-5, num_iter=config.NUM_EPOCHS)

print("Model compiled")

# calculate steps per epoch for training and val set
trainSteps = len(training_generator) // config.BATCH_SIZE
valSteps = len(valid_generator) // config.BATCH_SIZE

# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss": []}

best_val_loss = float('inf')

# initialize paths to save the model and plot
plot_path = os.path.join(config.BASE_OUTPUT, "plot.png")
model_path = os.path.join(config.BASE_OUTPUT, "model.pth")

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
    # set the model in training mode
    model.train()
    # initialize the total training and validation loss
    totalTrainLoss, totalValLoss = 0, 0
    # loop over the training set
    for (i, (x, y)) in enumerate(training_generator):  # TRAINING
        # send the input to the device
        (x, y) = (x.to(config.DEVICE, dtype=torch.double),
                  y.to(config.DEVICE, dtype=torch.long))
        print(x.dtype, y.dtype)
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFunc(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        # add the loss to the total training loss so far
        totalTrainLoss += loss

    # VALIDATION
    with torch.no_grad():  # switch off autograd
        # set the model in evaluation mode
        model.eval()
        # loop over the validation set
        for (x, y) in valid_generator:
            # send the input to the device
            (x, y) = (x.to(config.DEVICE, dtype=torch.double),
                      y.to(config.DEVICE, dtype=torch.long))
            # make the predictions and calculate the validation loss
            pred = model(x)
            totalValLoss += lossFunc(pred, y)

    # calculate the average training and validation loss
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    # update our training history
    H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    H["val_loss"].append(avgValLoss.cpu().detach().numpy())

    # check if current validation loss is less than the best validation loss
    if avgValLoss < best_val_loss:
        best_val_loss = avgValLoss
        # save the current model
        torch.save(model.state_dict(), model_path)

    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
    print("Train loss: {:.6f}, Val loss: {:.4f}".format(
        avgTrainLoss, avgValLoss))

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))


# load the best model
model.load_state_dict(torch.load(model_path))

# initialize a dictionary to store testing history
H = {"test_loss": [], "test_dice_et": [],
     "test_dice_tc": [], "test_dice_wt": []}
test_dice_et, test_dice_wt, test_dice_tc = 0, 0, 0

# TESTING
with torch.no_grad():  # switch off autograd
    # set the model in evaluation mode
    model.eval()
    # initialize the total test loss
    totalTestLoss = 0
    # loop over the test set
    for (x, y) in test_generator:
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
        # make the predictions and calculate the test loss
        pred = model(x)
        totalTestLoss += lossFunc(pred, y)
        test_dice_et += dice_coef_enhancing(y, pred)
        test_dice_tc += dice_coef_tc(y, pred)
        test_dice_wt += dice_coef_wt(y, pred)

    # calculate the average test loss
    avgTestLoss = totalTestLoss / len(test_generator)
    avg_test_dice_et = test_dice_et / len(test_generator)
    avg_test_dice_tc = test_dice_tc / len(test_generator)
    avg_test_dice_wt = test_dice_wt / len(test_generator)

    # update our training history
    H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
    H["test_dice_et"].append(avg_test_dice_et.cpu().detach().numpy())
    H["test_dice_tc"].append(avg_test_dice_tc.cpu().detach().numpy())
    H["test_dice_wt"].append(avg_test_dice_wt.cpu().detach().numpy())

    # print the model testing information
    print("Test loss: {:.4f}".format(avgTestLoss))
    print("Test dice et: {:.4f}".format(avg_test_dice_et))
    print("Test dice tc: {:.4f}".format(avg_test_dice_tc))
    print("Test dice wt: {:.4f}".format(avg_test_dice_wt))


# PLOT TRAINING LOSS
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="Train loss")
plt.plot(H["val_loss"], label="Val loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(plot_path)

# PLOT RESULTS
# Get the first batch of test data
x, _ = next(iter(test_generator))
x = x.to(config.DEVICE)

# Make prediction
prediction = model(x)

# Convert the prediction to numpy array and take the first image
X_test = test_generator.__getitem__(index)[0][0]
y_test = test_generator.__getitem__(index)[1][0]
flatten_y_test = np.argmax(y_test, axis=-1)
y_pred = model(X_test.unsqueeze(0).to(config.DEVICE)
               ).squeeze().cpu().detach().numpy()

# Plot the results
plt.figure(figsize=(12, 8))
plt.subplot(121), plt.plot(
    y_test[n_slice, :, :].flatten(), label='Ground truth')
plt.subplot(122), plt.plot(y_pred[n_slice, :, :].flatten(), label='Prediction')
