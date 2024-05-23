import torch
import torch.nn as nn
import torch.nn.init as init
import ChannelizedImage
from PatRecon import net
import gc
#import time


# Define the model class
class SPECT_Model_channelized2(nn.Module):
    def __init__(self, channelized=False, go_to_2x2=False, go_to_1x1=False):
        super().__init__()

        self.channelized = channelized
        self.go_to_2x2 = go_to_2x2 or go_to_1x1
        self.go_to_1x1 = go_to_1x1

        # make the layers.
        in_channels = 1
        out_channels1 = 2  # the number of channels out of the first layer
        out_channels = out_channels1 // 2

        #
        # Convolutional layers
        #

        # group 1: Input is [in_channels, 256, 256]; output is [out_channels1, 128, 128]
        self.conv_layer1 = net._make_layers(in_channels, out_channels1, 'conv4_s2', False)
        self.conv_layer2 = net._make_layers(out_channels1, out_channels1, 'conv3_s1', '2d')
        self.relu1 = nn.ReLU(inplace=True)

        # group 2: Input is [out_channels1, 128, 128]; output is [2out_channels1, 64, 64]
        self.conv_layer3 = net._make_layers(out_channels1, 2*out_channels1, 'conv4_s2', '2d', 'relu')
        self.conv_layer4 = net._make_layers(2*out_channels1, 2*out_channels1, 'conv3_s1', '2d')
        self.relu2 = nn.ReLU(inplace=True)

        # group 3: Input is [2out_channels1, 64, 64]; output is [4out_channels1, 32, 32]
        self.conv_layer5 = net._make_layers(2*out_channels1, 4*out_channels1, 'conv4_s2', '2d', 'relu')
        self.conv_layer6 = net._make_layers(4*out_channels1, 4*out_channels1, 'conv3_s1', '2d')
        self.relu3 = nn.ReLU(inplace=True)

        # group 4: Input is [4out_channels1, 32, 32]; output is [8out_channels1, 16, 16]
        self.conv_layer7 = net._make_layers(4*out_channels1, 8*out_channels1, 'conv4_s2', '2d', 'relu')
        self.conv_layer8 = net._make_layers(8*out_channels1, 8*out_channels1, 'conv3_s1', '2d')
        self.relu4 = nn.ReLU(inplace=True)

        # group 5: Input is [8out_channels1, 16, 16]; output is [16out_channels1, 8, 8]
        self.conv_layer9 = net._make_layers(8 * out_channels1, 16 * out_channels1, 'conv4_s2', '2d', 'relu')
        self.conv_layer10 = net._make_layers(16 * out_channels1, 16 * out_channels1, 'conv3_s1', '2d')
        self.relu5 = nn.ReLU(inplace=True)

        # group 6: Input is [16out_channels1, 8, 8]; output is [32out_channels1, 4, 4]
        self.conv_layer11 = net._make_layers(16 * out_channels1, 32 * out_channels1, 'conv4_s2', '2d', 'relu')
        self.conv_layer12 = net._make_layers(32 * out_channels1, 32 * out_channels1, 'conv3_s1', '2d')
        self.relu6 = nn.ReLU(inplace=True)

        if self.go_to_2x2:
            # group 7: Input is [32out_channels1, 4, 4]; output is [64out_channels1, 2, 2]
            self.conv_layer13 = net._make_layers(32 * out_channels1, 64 * out_channels1, 'conv4_s2', '2d', 'relu')
            self.conv_layer14 = net._make_layers(64 * out_channels1, 64 * out_channels1, 'conv3_s1', '2d')
            self.relu7 = nn.ReLU(inplace=True)

            if self.go_to_1x1:
                # group 8: Input is [64out_channels1, 2, 2]; output is [128out_channels1, 1, 1]
                self.conv_layer15 = net._make_layers(64 * out_channels1, 128 * out_channels1, 'conv4_s2', '2d', 'relu')
                self.conv_layer16 = net._make_layers(128 * out_channels1, 128 * out_channels1, 'conv3_s1', '2d')
                self.relu8 = nn.ReLU(inplace=True)

                #
                # Transitional layers: [128*out_channels1, 1, 1]
                #
                self.trans_layer1 = net._make_layers(128 * out_channels1, 128*out_channels1, 'conv1_s1', False, 'relu')
                self.trans_layer2 = net._make_layers(128*out_channels1, 128*out_channels1, 'deconv1_s1', False, 'relu')

                #
                # De-convolutional layers
                #

                # group 9:
                self.deconv_layer16 = net._make_layers(256*out_channels, 128*out_channels, 'deconv4_s2', '2d', 'relu')
                self.deconv_layer15 = net._make_layers(128*out_channels, 128*out_channels, 'deconv3_s1', '2d', 'relu')
            else:
                #
                # Transitional layers
                #
                self.trans_layer1 = net._make_layers(64 * out_channels1, 64 * out_channels1, 'conv1_s1', False, 'relu')
                self.trans_layer2 = net._make_layers(64 * out_channels1, 64 * out_channels1, 'deconv1_s1', False, 'relu')

                #
                # De-convolutional layers
                #

            # group 8:
            self.deconv_layer14 = net._make_layers(128 * out_channels, 64 * out_channels, 'deconv4_s2', '2d', 'relu')
            self.deconv_layer13 = net._make_layers(64 * out_channels, 64 * out_channels, 'deconv3_s1', '2d', 'relu')

        else:
            # transitional layers for 4x4
            self.trans_layer1 = net._make_layers(32 * out_channels1, 32 * out_channels1, 'conv1_s1', False, 'relu')
            self.trans_layer2 = net._make_layers(32 * out_channels1, 32 * out_channels1, 'deconv1_s1', False, 'relu')

        # group 6:
        self.deconv_layer12 = net._make_layers(64*out_channels, 32*out_channels, 'deconv4_s2', '2d', 'relu')
        self.deconv_layer11 = net._make_layers(32*out_channels, 32*out_channels, 'deconv3_s1', '2d', 'relu')

        # group 5:
        self.deconv_layer10 = net._make_layers(32*out_channels, 16*out_channels, 'deconv4_s2', '2d', 'relu')
        self.deconv_layer9 = net._make_layers(16*out_channels, 16*out_channels, 'deconv3_s1', '2d', 'relu')

        # group 4:
        self.deconv_layer8 = net._make_layers(16*out_channels, 8*out_channels, 'deconv4_s2', '2d', 'relu')
        self.deconv_layer7 = net._make_layers(8*out_channels, 8*out_channels, 'deconv3_s1', '2d', 'relu')

        # group 3:
        self.deconv_layer6 = net._make_layers(8*out_channels, 4*out_channels, 'deconv4_s2', '2d', 'relu')
        self.deconv_layer5 = net._make_layers(4*out_channels, 4*out_channels, 'deconv3_s1', '2d', 'relu')

        # group 2:
        self.deconv_layer4 = net._make_layers(4*out_channels, 2*out_channels, 'deconv4_s2', '2d', 'relu')
        self.deconv_layer3 = net._make_layers(2*out_channels, 2*out_channels, 'deconv3_s1', '2d', 'relu')

        # group 1:
        self.deconv_layer2 = net._make_layers(2*out_channels, out_channels, 'deconv4_s2', '2d', 'relu')
        self.deconv_layer1 = net._make_layers(out_channels, out_channels, 'deconv3_s1', '2d', 'relu')

        # output
        self.output_layer = net._make_layers(out_channels, 1, 'conv1_s1', False) \
            if self.channelized else None

        # network initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Initialize weights using Xavier uniform initialization
                init.xavier_uniform_(m.weight)
                # Initialize biases to zeros
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # X is [1, 256, 256]

        # clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # get the number of channels
        num_channels = x.shape[0]

        # representation network
        # group 1
        conv1 = self.conv_layer1(x)
        conv2 = self.conv_layer2(conv1)
        relu1 = self.relu1(conv1 + conv2)  # [out_channels1, 128, 128]

        # group 2
        conv3 = self.conv_layer3(relu1)
        conv4 = self.conv_layer4(conv3)
        relu2 = self.relu2(conv3 + conv4)  # [2*out_channels1, 64, 64]

        # group 3
        conv5 = self.conv_layer5(relu2)
        conv6 = self.conv_layer6(conv5)
        relu3 = self.relu2(conv5 + conv6)  # [4*out_channels1, 32, 32]

        # group 4
        conv7 = self.conv_layer7(relu3)
        conv8 = self.conv_layer8(conv7)
        relu4 = self.relu4(conv7 + conv8)  # [8*out_channels1, 16, 16]

        # group 5
        conv9 = self.conv_layer9(relu4)
        conv10 = self.conv_layer10(conv9)
        relu5 = self.relu5(conv9 + conv10)  # [16*out_channels1, 8, 8]

        # group 6
        conv11 = self.conv_layer11(relu5)
        conv12 = self.conv_layer12(conv11)
        relu6 = self.relu6(conv11 + conv12)  # [32*out_channels1, 4, 4]

        deconv13 = None
        deconv15 = None
        if self.go_to_2x2:
            # group 7
            conv13 = self.conv_layer13(relu6)
            conv14 = self.conv_layer14(conv13)
            relu7 = self.relu7(conv13 + conv14)  # [64*out_channels1, 2, 2]

            if self.go_to_1x1:
                # group 8
                conv15 = self.conv_layer15(relu7)
                conv16 = self.conv_layer16(conv15)
                relu8 = self.relu8(conv15 + conv16)  # [128*out_channels1, 1, 1]

                # transition
                trans1 = self.trans_layer1(relu8)
                trans2 = self.trans_layer2(trans1)

                # De-convolution
                deconv16 = self.deconv_layer16(trans2)  # [256, 1, 1]
                deconv15 = self.deconv_layer15(deconv16)  # [256, 1, 1]
            else:
                # transition
                trans1 = self.trans_layer1(relu7)
                trans2 = self.trans_layer2(trans1)  # [64*out_channels1, 2, 2]
                deconv15 = trans2

            deconv14 = self.deconv_layer14(deconv15)  # [32*out_channels1, 4, 4]
            deconv13 = self.deconv_layer13(deconv14)  # [32*out_channels1, 4, 4]
        else:
            # transition: [32*out_channels1, 4, 4]
            trans1 = self.trans_layer1(relu6)
            trans2 = self.trans_layer2(trans1)
            deconv13 = trans2

        deconv12 = self.deconv_layer12(deconv13)  # [32*out_channels, 8, 8]
        deconv11 = self.deconv_layer11(deconv12)  # [32*out_channels, 8, 8]

        deconv10 = self.deconv_layer10(deconv11)  # [16*out_channels, 16, 16]
        deconv9 = self.deconv_layer9(deconv10)    # [16*out_channels, 16, 16]

        deconv8 = self.deconv_layer8(deconv9)     # [8*out_channels, 32, 32]
        deconv7 = self.deconv_layer7(deconv8)     # [8*out_channels, 32, 32]

        deconv6 = self.deconv_layer6(deconv7)     # [4*out_channels, 64, 64]
        deconv5 = self.deconv_layer5(deconv6)     # [4*out_channels, 64, 64]

        deconv4 = self.deconv_layer4(deconv5)     # [2*out_channels, 128, 128]
        deconv3 = self.deconv_layer3(deconv4)     # [2*out_channels, 128, 128]

        deconv2 = self.deconv_layer2(deconv3)     # [1*out_channels, 256, 256]
        deconv1 = self.deconv_layer1(deconv2)     # [1*out_channels, 256, 256]

        ### output
        result = deconv1
        if self.channelized:
            #start_time = time.time()
            # extract the lowest-level of the result
            result_c = torch.zeros(num_channels, 256, 256, device=result.device)

            if self.go_to_1x1:
                # level = 0
                result_c.view(-1, 1, 256 ** 2)[:, 0, 0] = torch.mean(trans2, dim=1).view(-1, 1, 1)[:, 0, 0]

                last_i = 1
                dimen = 2
                size_i = 3 * dimen**2 // 4
                first_i, last_i = last_i, last_i + size_i
                result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv15, dim=1).view(-1, 1, dimen ** 2)[:, 0,
                                                                       0:size_i]
                dimen = 4
                size_i = 3 * dimen**2 // 4
                first_i, last_i = last_i, last_i + size_i
                result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv13, dim=1).view(-1, 1, dimen ** 2)[:, 0,
                                                                       0:size_i]

            elif self.go_to_2x2:
                tmp_tensor = ChannelizedImage.channelize(torch.mean(deconv15, dim=1))  # 2x2
                dimen = 2
                size_i = 2**2
                first_i, last_i = 0, size_i
                result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = tmp_tensor.view(-1, 1, dimen ** 2)[:, 0, 0:size_i]

                # extract the rest of the result
                last_i = 4
                dimen = 4
                size_i = 3 * dimen**2 // 4
                first_i, last_i = last_i, last_i + size_i
                result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv13, dim=1).view(-1, 1, dimen ** 2)[:, 0,
                                                                     0:size_i]
            else:
                last_i = 0
                dimen = 4
                size_i = 16
                first_i, last_i = last_i, last_i + size_i
                result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv13, dim=1).view(-1, 1, dimen ** 2)[:,
                                                                       0, 0:size_i]

            # extract the rest of the result
            first_i, last_i = 0, 16  # initialized as if used in previous layer
            # level = 3   # requires 48 values;  indices3 = ChannelizedImage.get_indices(256, 3)
            dimen = 8
            size_i = 3 * dimen**2 // 4
            first_i, last_i = last_i, last_i + size_i
            result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv11, dim=1).view(-1, 1, dimen ** 2)[:, 0,
                                                                 0:size_i]

            # level = 4   # requires 192 values;  indices4 = ChannelizedImage.get_indices(256, 4)
            dimen *= 2
            size_i = 3 * dimen**2 // 4
            first_i, last_i = last_i, last_i + size_i
            result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv9, dim=1).view(-1, 1, dimen ** 2)[:, 0,
                                                                 0:size_i]

            # level = 5   # requires 768 values;  indices5 = ChannelizedImage.get_indices(256, 5)
            dimen *= 2
            size_i = 3 * dimen**2 // 4
            first_i, last_i = last_i, last_i + size_i
            result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv7, dim=1).view(-1, 1, dimen ** 2)[:, 0,
                                                                 0:size_i]

            # level = 6   # requires 3072 values;  indices6 = ChannelizedImage.get_indices(256, 6)
            dimen *= 2
            size_i = 3 * dimen**2 // 4
            first_i, last_i = last_i, last_i + size_i
            result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv5, dim=1).view(-1, 1, dimen ** 2)[:, 0,
                                                                 0:size_i]

            # level = 7   # requires 12288 values;  indices7 = ChannelizedImage.get_indices(256, 7)
            dimen *= 2
            size_i = 3 * dimen**2 // 4
            first_i, last_i = last_i, last_i + size_i
            result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv3, dim=1).view(-1, 1, dimen ** 2)[:, 0,
                                                                 0:size_i]

            # level = 8   # requires 49152 values;  indices8 = ChannelizedImage.get_indices(256, 8)
            dimen *= 2
            size_i = 3 * dimen**2 // 4
            first_i, last_i = last_i, last_i + size_i
            result_c.view(-1, 1, 256 ** 2)[:, 0, first_i:last_i] = torch.mean(deconv1, dim=1).view(-1, 1, dimen ** 2)[:, 0,
                                                                 0:size_i]

            # need to de-channelize and get an extra dimension
            #mid_time = time.time()
            result = ChannelizedImage.dechannelize(result_c).unsqueeze(1)
            #end_time = time.time()
            #print(f"Channelize time: {mid_time-start_time}; {end_time-start_time}")

        # return result
        return result

    def load(self, filename):  #, device):
        #checkpoint = torch.load(filename, map_location=device)
        checkpoint = torch.load(filename) 
        self.load_state_dict(checkpoint['state_dict'])

    def load_and_replace(self, filename, device):  #, device):
        """
        This loads the model, checks model-dependent variables, and then fills the model
        """
        #checkpoint = torch.load(filename, map_location=device)
        checkpoint = torch.load(filename, map_location = device)
        try:
            channelized = checkpoint['channelized']
        except KeyError:
            channelized = True
            print("Unable to find channelization state during reload. Assuming true.")
        go_to_2x2 = checkpoint['go_to_2x2']
        go_to_1x1 = checkpoint['go_to_1x1']
        if self.go_to_2x2 != go_to_2x2:
            print(f"Parameter 'go_to_2x2' from disk does not match current. current = {self.go_to_2x2}.")
        if self.go_to_1x1 != go_to_1x1:
            print(f"Parameter 'go_to_1x1' from disk does not match current. current = {self.go_to_1x1}.")

        # create the new
        new_model = SPECT_Model_channelized2(channelized, self.go_to_2x2, self.go_to_1x1)
        try:
            new_model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print("Exception occurred during model initialization:", e)
            print("Will try to continue.")


        # return
        return new_model

    def save(self, filename):
        # save the state
        state = {
            'state_dict': self.state_dict(),
            'channelized': self.channelized,
            'go_to_2x2': self.go_to_2x2,
            'go_to_1x1': self.go_to_1x1
        }
        torch.save(state, filename)

