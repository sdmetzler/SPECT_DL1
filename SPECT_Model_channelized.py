import torch
import torch.nn as nn
import torch.nn.functional as nnf

import SPECT_Dataset_Channels
from PatRecon import net
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import gc
from memory_profiler import profile
#from torchviz import make_dot


# Define the model class
def fluff(tensor, final_size):
    # Convert the input to a PyTorch tensor if it's not already
    if not isinstance(tensor, torch.Tensor):
        # If not, convert it to a PyTorch tensor
        tensor = torch.tensor(tensor)

    # Compute the size of the input tensor
    instance = tensor.shape[0]

    # Compute the number of times to replicate along each dimension
    size = tensor.shape[1]
    num = final_size // size

    # Reshape and broadcast the input tensor
    # [instance, 2, 2, 1, 1] after unsqueezes
    # [instance, 2, 256, 128, 1] after repeat;
    replicated_average = tensor.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, num, 1, num)

    # Reshape the result to the final size
    result = replicated_average.view(-1, final_size, final_size)

    return result

def extract(tensor):
    # Convert the input to a PyTorch tensor if it's not already
    if not isinstance(tensor, torch.Tensor):
        # If not, convert it to a PyTorch tensor
        tensor = torch.tensor(tensor)

    # take the tensor and average over the first two non-instance dimensions
    return fluff(torch.mean(tensor, dim=(1, 2)).squeeze(), 256)


def extract_first_layers_2(tensor):  # [instance,:, :, 2, 2]
    # Convert the input to a PyTorch tensor if it's not already
    if not isinstance(tensor, torch.Tensor):
        # If not, convert it to a PyTorch tensor
        tensor = torch.tensor(tensor)

    # get the average over the channels
    means_2x2 = torch.mean(tensor, dim=(1, 2))  # [instance, 2, 2]

    # first result is the average of the 4x4 grids
    instances = means_2x2.shape[0]

    # expand to full size
    result0 = fluff(torch.mean(means_2x2, dim=(1, 2)).view(-1, 1, 1), 256)
    result1 = fluff(means_2x2, 256)

    # second result is to subtract off the average from the previous
    result1 -= result0

    # put the layers together
    return torch.cat((result0.unsqueeze(1), result1.unsqueeze(1)), dim=1)

def extract_first_layers_3(tensor):  # [instance,:, :, 4, 4]
    # Convert the input to a PyTorch tensor if it's not already
    if not isinstance(tensor, torch.Tensor):
        # If not, convert it to a PyTorch tensor
        tensor = torch.tensor(tensor)

    # get the average over the channels
    means_4x4 = torch.mean(tensor, dim=(1, 2))  # [instance, 4, 4]

    # first result is the average of the 4x4 grids
    instances = means_4x4.shape[0]

    # Compute the means for each 2x2 region
    means_2x2 = torch.mean(means_4x4.view(instances, 2, 2, 2, 2), dim=(2, 4))

    # expand to full size
    result0 = fluff(torch.mean(means_2x2, dim=(1, 2)).view(-1, 1, 1), 256)
    result1 = fluff(means_2x2, 256)
    result2 = fluff(means_4x4, 256)

    # second result is to subtract off the average from the previous
    result2 -= result1
    result1 -= result0

    # put the layers together
    return torch.cat((result0.unsqueeze(1), result1.unsqueeze(1), result2.unsqueeze(1)), dim=1)


class SPECT_Model_channelized(nn.Module):
    def __init__(self, extract_at_end_only=False, go_to_2x2=False, go_to_1x1=False):
        super().__init__()

        in_channels = 1
        out_channels = 1 # the highest-resolution channel; the rest are saved along the way
        self.extract_at_end_only = extract_at_end_only
        self.go_to_2x2 = go_to_2x2 or go_to_1x1
        self.go_to_1x1 = go_to_1x1
        print(f"Module constructed: {self.extract_at_end_only} {self.go_to_2x2} {self.go_to_1x1}")
        self.conv_layer1 = net._make_layers(in_channels, 256, 'conv4_s2', False)
        self.conv_layer2 = net._make_layers(256, 256, 'conv3_s1', '2d')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_layer3 = net._make_layers(256, 512, 'conv4_s2', '2d', 'relu')
        self.conv_layer4 = net._make_layers(512, 512, 'conv3_s1', '2d')
        self.relu4 = nn.ReLU(inplace=True)
        self.conv_layer5 = net._make_layers(512, 1024, 'conv4_s2', '2d', 'relu')
        self.conv_layer6 = net._make_layers(1024, 1024, 'conv3_s1', '2d')
        self.relu6 = nn.ReLU(inplace=True)
        self.conv_layer7 = net._make_layers(1024, 2048, 'conv4_s2', '2d', 'relu')
        self.conv_layer8 = net._make_layers(2048, 2048, 'conv3_s1', '2d')
        self.relu8 = nn.ReLU(inplace=True)
        self.conv_layer9 = net._make_layers(2048, 4096, 'conv4_s2', '2d', 'relu')
        self.conv_layer10 = net._make_layers(4096, 4096, 'conv3_s1', '2d')
        self.relu10 = nn.ReLU(inplace=True)
        # adding the following extraction layers to get it down to 1x1
        self.conv_layer11 = []
        self.conv_layer12 = []
        self.relu12 = []
        self.conv_layer13 = []
        self.conv_layer14 = []
        self.relu14 = []
        self.trans_layer1 = []
        self.trans_layer2 = []
        if self.go_to_2x2:
            self.conv_layer11 = net._make_layers(4096, 8192, 'conv4_s2', '2d', 'relu')
            self.conv_layer12 = net._make_layers(8192, 8192, 'conv3_s1', '2d')
            self.relu12 = nn.ReLU(inplace=True)
            if self.go_to_1x1:
                self.conv_layer13 = net._make_layers(8192, 16384, 'conv4_s2', '2d', 'relu')
                self.conv_layer14 = net._make_layers(16384, 16384, 'conv3_s1', '2d')
                self.relu14 = nn.ReLU(inplace=True)
                self.trans_layer1 = net._make_layers(16384, 16384, 'conv1_s1', False, 'relu')
                self.trans_layer2 = net._make_layers(8192, 8192, 'deconv1x1_s1', False, 'relu')
            else:
                assert self.go_to_2x2
                self.trans_layer1 = net._make_layers(8192, 8192, 'conv1_s1', False, 'relu')
                #self.trans_layer2 = net._make_layers(4096, 4096, 'deconv1x1_s1', False, 'relu')
                self.trans_layer2 = net._make_layers(8192, 4096, 'deconv4x4_s2', '3d', 'relu')
        else:
            assert not self.go_to_1x1
            assert not self.go_to_2x2
            ######### transform module
            self.trans_layer1 = net._make_layers(4096, 4096, 'conv1_s1', False, 'relu')
            self.trans_layer2 = net._make_layers(2048, 2048, 'deconv1x1_s1', False, 'relu')

        ######### generation network - deconvolution layers
        #self.deconv_layer10 = net._make_layers(2048, 1024, 'deconv4x4_s2', '3d', 'relu')
        #self.deconv_layer8 = net._make_layers(1024, 512, 'deconv4x4_s2', '3d', 'relu')
        # inserted the following layers before layer7 since it now goes down to 1x1
        self.deconv_layer13 = []
        self.deconv_layer12 = []
        self.deconv_layer11 = []
        self.deconv_layer10 = []
        self.deconv_layer9 = []
        self.deconv_layer8 = []
        if self.go_to_1x1:
            self.deconv_layer13 = net._make_layers(4096, 2096, 'deconv3x3_s1', '3d', 'relu')
            self.deconv_layer12 = net._make_layers(4096, 2048, 'deconv4x4_s2', '3d', 'relu')
        elif self.go_to_2x2:
            print("Inserting 2x2 layers")
            self.deconv_layer11 = net._make_layers(4096, 2048 , 'deconv3x3_s1', '3d', 'relu')
            self.deconv_layer10 = net._make_layers(2048, 1024, 'deconv4x4_s2', '3d', 'relu')
            self.deconv_layer9 = net._make_layers(1024, 1024, 'deconv3x3_s1', '3d', 'relu')
            self.deconv_layer8 = net._make_layers(1024, 512, 'deconv4x4_s2', '3d', 'relu')
        else:
            assert not self.go_to_1x1
            assert not self.go_to_2x2
            self.deconv_layer10 = net._make_layers(2048, 1024, 'deconv4x4_s2', '3d', 'relu')
            self.deconv_layer8 = net._make_layers(1024, 512, 'deconv4x4_s2', '3d', 'relu')

        self.deconv_layer7 = net._make_layers(512, 512, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer6 = net._make_layers(512, 256, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer5 = net._make_layers(256, 256, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer4 = net._make_layers(256, 128, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer3 = net._make_layers(128, 128, 'deconv3x3_s1', '3d', 'relu')
        self.deconv_layer2 = net._make_layers(128, 64, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layer1 = net._make_layers(64, 64, 'deconv3x3_s1', '3d', 'relu')

        # I tried to add a layer, but the computer ran out of memory
        ## scale up one more time since the previous gets the image to [64, 64, 128, 128]
        self.deconv_layerA = net._make_layers(64, 32, 'deconv4x4_s2', '3d', 'relu')
        self.deconv_layerB = net._make_layers(32, 32, 'deconv3x3_s1', '3d', 'relu')

        self.deconv_layer0 = net._make_layers(32, 1, 'conv1x1_s1', False, 'relu')
        self.output_layer = net._make_layers(2, out_channels, 'conv1_s1', False)

        # network initialization
        net._initialize_weights(self)

    @profile
    def forward(self, x):
        # clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        ### representation network
        # X is [1, 128, 256]
        print(f"Input size into conv_layer1 is {x.shape}.")
        conv1 = self.conv_layer1(x)  # [256, 64, 128]
        print(f"Input size into conv_layer2 is {conv1.shape}.")
        conv2 = self.conv_layer2(conv1)  # [256, 64, 128]
        relu2 = self.relu2(conv1 + conv2)  # [256, 64, 128]

        print(f"Input size into conv_layer3 is {relu2.shape}.")
        conv3 = self.conv_layer3(relu2)  # [512, 32, 64]
        print(f"Input size into conv_layer4 is {conv3.shape}.")
        conv4 = self.conv_layer4(conv3)  # [512, 32, 64]
        relu4 = nnf.max_pool2d(self.relu4(conv3 + conv4), (1, 2), (1, 2))  # [512, 32, 32]

        print(f"Input size into conv_layer5 is {relu4.shape}.")
        conv5 = self.conv_layer5(relu4)  # [1024, 16, 16]
        print(f"Input size into conv_layer6 is {conv5.shape}.")
        conv6 = self.conv_layer6(conv5)  # [1024, 16, 16]
        relu6 = self.relu6(conv5 + conv6)  # [1024, 16, 16]

        print(f"Input size into conv_layer7 is {relu6.shape}.")
        conv7 = self.conv_layer7(relu6)  # [2048, 8, 8]
        print(f"Input size into conv_layer8 is {conv7.shape}.")
        conv8 = self.conv_layer8(conv7)  # [2048, 8, 8]
        relu8 = self.relu8(conv7 + conv8)  # [2048, 8, 8]

        print(f"Input size into conv_layer9 is {relu8.shape}.")
        conv9 = self.conv_layer9(relu8)  # [4096, 4, 4]
        print(f"Input size into conv_layer10 is {conv9.shape}.")
        conv10 = self.conv_layer10(conv9)  # [4096, 4, 4]
        relu10 = self.relu10(conv9 + conv10)  # [4096, 4, 4]

        """
        conv1 = []
        conv2 = []
        conv3 = []
        conv4 = []
        conv5 = []
        conv6 = []
        conv7 = []
        conv8 = []
        conv9 = []
        conv10 = []
        relu2 = []
        relu4 = []
        relu6 = []
        relu8 = []
        """
        
        # allocate for de-convolution
        deconv13 = []
        deconv12 = []
        deconv11 = []
        deconv10 = []
        deconv9 = []
        deconv8 = []
        trans_features = []
        if self.go_to_2x2:
            print("Executing 2x2 layers")
            print(f"Input size into conv_layer11 is {relu10.shape}.")
            conv11 = self.conv_layer11(relu10)  # [8192, 2, 2]
            print(f"Input size into conv_layer12 is {conv11.shape}.")
            conv12 = self.conv_layer12(conv11)  # [8192, 2, 2]
            relu12 = self.relu10(conv11 + conv12)  # [8192, 2, 2]

            if self.go_to_1x1:
                conv13 = self.conv_layer13(relu12)  # [16384, 1, 1]
                conv14 = self.conv_layer14(conv13)  # [16384, 1, 1]
                relu14 = self.relu10(conv13 + conv14)  # [16384, 1, 1]

                ### transform module
                features = self.trans_layer1(relu14)
                trans_features = features.view(-1, 8192, 2, 1, 1)
                trans_features = self.trans_layer2(trans_features)  # [8192, 2, 1, 1]
                deconv13 = self.deconv_layer13(trans_features)  # [1024, 4, 8, 8]
                deconv12 = self.deconv_layer12(deconv13)  # [1024, 4, 8, 8]
                deconv11 = self.deconv_layer11(deconv12)  # [1024, 4, 8, 8]
            else:
                assert self.go_to_2x2
                ### transform module
                print(f"Input size into trans_layer1 is {relu12.shape}.")
                trans_features = self.trans_layer1(relu12)  # [8192, 2, 2]
                trans_features = trans_features.view(-1, 8192, 1, 2, 2)
                print(f"Input size into trans_layer2 is {trans_features.shape}.")
                trans_features1 = self.trans_layer2(trans_features)  # [4096, 2, 4, 4]
                print(f"Input size into deconv11 is {trans_features1.shape}.")
                deconv11 = self.deconv_layer11(trans_features1)  # [2048, 2, 4, 4]

            print(f"Input size into deconv10 is {deconv11.shape}.")
            deconv10 = self.deconv_layer10(deconv11)  # [1024, 4, 8, 8]
            print(f"Input size into deconv9 is {deconv10.shape}.")
            deconv9 = self.deconv_layer9(deconv10)  # [1024, 4, 8, 8]
            print(f"Input size into deconv8 is {deconv9.shape}.")
            deconv8 = self.deconv_layer8(deconv9)  # [512, 8, 16, 16]
        else:
            ### transform module
            features = self.trans_layer1(relu10)
            trans_features = features.view(-1, 2048, 2, 4, 4)
            trans_features = self.trans_layer2(trans_features)  # [2048, 2, 4, 4]

            ### generation network
            deconv10 = self.deconv_layer10(trans_features)  # [1024, 4, 8, 8]
            deconv8 = self.deconv_layer8(deconv10)  # [512, 8, 16, 16]

        # continue with the rest
        print(f"Input size into deconv7 is {deconv8.shape}.")
        deconv7 = self.deconv_layer7(deconv8)  # [512, 8, 16, 16]
        print(f"Input size into deconv6 is {deconv7.shape}.")
        deconv6 = self.deconv_layer6(deconv7)  # [256, 16, 32, 32]
        deconv5 = self.deconv_layer5(deconv6)  # [256, 16, 32, 32]
        deconv4 = self.deconv_layer4(deconv5)  # [128, 32, 64, 64]
        deconv3 = self.deconv_layer3(deconv4)  # [128, 32, 64, 64]
        deconv2 = self.deconv_layer2(deconv3)  # [64, 64, 128, 128]
        deconv1 = self.deconv_layer1(deconv2)  # [64, 64, 128, 128]

        new_tensor = torch.mean(deconv1, dim=2).unsqueeze(2)
        deconvA = self.deconv_layerA(new_tensor)  # [32, 2, 256, 256]
        deconvB = self.deconv_layerB(deconvA)  # [32, 2, 256, 256]

        ### output
        out = self.deconv_layer0(deconvB)  # [1, 2, 256, 256]
        out = torch.squeeze(out, 1)  # [2, 256, 256]
        out = self.output_layer(out)  # [1, 256, 256]
        #out = nnf.interpolate(out, size=(256, 256), mode='bicubic')

        # extract the result
        result = []
        if not self.extract_at_end_only:
            print("Extracting layers")
            # extract the first two channels
            #result = extract_first_layers(trans_features)  # [instances, 2, 256, 256]
            if self.go_to_1x1:
                result = extract(trans_features).unsqueeze(1)  # [instances, 1, 256, 256]
                result = torch.cat((result, extract(deconv13).unsqueeze(1)), dim=1)  # [instances, 2, 256, 256]
                result = torch.cat((result, extract(deconv11).unsqueeze(1)), dim=1)  # [instances, 3, 256, 256]
                result = torch.cat((result, extract(deconv9).unsqueeze(1)), dim=1)  # [instances, 4, 256, 256]
            elif self.go_to_2x2:
                result = extract_first_layers_2(trans_features)  # [instances, 2, 256, 256]
                result = torch.cat((result, extract(deconv11).unsqueeze(1)), dim=1)  # [instances, 3, 256, 256]
                result = torch.cat((result, extract(deconv9).unsqueeze(1)), dim=1)  # [instances, 4, 256, 256]
            else:
                result = extract_first_layers_3(trans_features)  # [instances, 3, 256, 256]
                result = torch.cat((result, extract(deconv10).unsqueeze(1)), dim=1)  # [instances, 4, 256, 256]
            result = torch.cat((result, extract(deconv7).unsqueeze(1)), dim=1)  # [instances, 5, 256, 256]
            result = torch.cat((result, extract(deconv5).unsqueeze(1)), dim=1)  # [instances, 6, 256, 256]
            result = torch.cat((result, extract(deconv3).unsqueeze(1)), dim=1)  # [instances, 7, 256, 256]
            result = torch.cat((result, extract(deconv1).unsqueeze(1)), dim=1)  # [instances, 8, 256, 256]
            result = torch.cat((result, out), dim=1)  # [instances, 9, 256, 256]
        else:
            print("Extracting at end")
            result = SPECT_Dataset_Channels.channelize_tensor(out)

        # return result
        return result

    def load(self, filename):  #, device):
        #checkpoint = torch.load(filename, map_location=device)
        checkpoint = torch.load(filename) 
        self.load_state_dict(checkpoint['state_dict'])

    def load_and_replace(self, filename):  #, device):
        """
        This loads the model, checks model-dependent variables, and then fills the model
        """
        #checkpoint = torch.load(filename, map_location=device)
        checkpoint = torch.load(filename)
        extract_only = checkpoint['extract_only']
        go_to_2x2 = checkpoint['go_to_2x2']
        go_to_1x1 = checkpoint['go_to_1x1']

        # create the new
        new_model = SPECT_Model_channelized(extract_only, go_to_2x2, go_to_1x1)
        new_model.load_state_dict(checkpoint['state_dict'])

        # return
        return new_model

    def save(self, filename):
        # save the state
        state = {
            'state_dict': self.state_dict(),
            'extract_only': self.extract_at_end_only,
            'go_to_2x2': self.go_to_2x2,
            'go_to_1x1': self.go_to_1x1
        }
        torch.save(state, filename)

