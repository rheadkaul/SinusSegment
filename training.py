from load_dataset import load_data
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from UnetPlusPlus import UnetPlusPlus
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

# channel_input, channel_output, learning rate, number of stack
c_in, c_out, lr, stack = 1, 1, 0.0001, 3
# stack = 3 * 2 +1,
model = UnetPlusPlus(stack=stack * 2 + 1, num_classes=1, deep_supervision=True).to(device=device, dtype=dtype)

# create Adam optimizer
optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), eps=1e-4)

# loading data
train_loader = load_data(True, stack)
epochs = 100
batch_size = 4

for epoch in range(epochs + 1):

    total_loss = 0
    # training
    for k, (inputs, labels, name) in tqdm(enumerate(train_loader, 1)):

        # dimension: (1,z,x,y) -> (z,1,x,y)
        labels = labels.permute(1, 0, 2, 3)
        # dimension: (1,z+stack,h,w) -> (z+stack,h,w)
        inputs = inputs.squeeze(0)

        # creating sliding window for stacking slices
        window_size = stack * 2 + 1
        stride = 1
        unfolded_input = inputs.unfold(0, window_size, stride)
        # [z, x, y, window_size] -> [z, window_size, x, y]
        unfolded_input = unfolded_input.permute(0, 3, 1, 2)

        # get number of batches for this volume
        num_batches = math.ceil(unfolded_input.shape[0] / batch_size)

        for i in range(num_batches):
            inputs_batch = unfolded_input[i * batch_size: (i + 1) * batch_size]  # 每个batch取4个
            labels_batch = labels[i * batch_size: (i + 1) * batch_size]

            if len(labels_batch.shape) == 3:
                # (1,h,w) -> (1,1,h,w)
                labels_batch = labels_batch.unsqueeze(0)

            inputs_batch, labels_batch = inputs_batch.to(device=device, dtype=dtype), labels_batch.to(device=device,
                                                                                                      dtype=dtype)
            output_list = model(inputs_batch)

            loss = 0
            for j in range(len(output_list)):
                loss += torch.nn.functional.mse_loss(output_list[j], labels_batch)

            optim.zero_grad()
            loss.backward()
            optim.step()

            print('loss: {}'.format(loss.item()))

    if epoch % 5 == 0 or epoch == 100:
        state = {
            'state': model.state_dict(),
            'epoch': epoch
        }
        torch.save(state, './parameters/epoch_' + str(epoch) + '.pth')
        print('saving model')
