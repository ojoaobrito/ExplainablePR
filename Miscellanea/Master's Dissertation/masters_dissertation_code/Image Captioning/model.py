import sys
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, model_type, mode):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        '''super(EncoderCNN, self).__init__()
        
        if(model_type == "resnet_152"): model = models.resnet152(pretrained=False)
        else: model = models.vgg19(pretrained=True)
        
        modules = list(model.children())[:-1]      # delete the last fc layer.
        self.model = nn.Sequential(*modules)
        
        if(model_type == "resnet_152"): self.linear = nn.Linear(model.fc.in_features, embed_size)
        else: self.linear = nn.Linear(model.classifier[0].in_features, embed_size)
        
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)'''

        super(EncoderCNN, self).__init__()
        self.mode = mode
        self.model_type = model_type
        #model = models.vgg19(pretrained=True)
        model = models.resnet152(pretrained=True)

        # ----------------------------------------------
        # feature extractor
        # ----------------------------------------------
        modules = list(model.children())[:-1] 
        self.feature_extractor = nn.Sequential(*modules)

        # ---------------------------------------------------------------------------------
        # linear transformations
        # ---------------------------------------------------------------------------------
        if(self.model_type == "vgg_19"):
            self.linear_to_embed_size = nn.Linear(model.classifier[0].in_features, embed_size)
            self.linear_to_embed_size_concat = nn.Linear(model.classifier[0].in_features * 2, embed_size)

        elif(self.model_type == "resnet_152"):
            self.linear_to_embed_size = nn.Linear(model.fc.in_features, embed_size)
            self.linear_to_embed_size_concat = nn.Linear(model.fc.in_features * 2, embed_size)

        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        '''with torch.no_grad():
            features = self.model(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features'''

        if(self.mode == "side_by_side"):
            with torch.no_grad():
                features = self.feature_extractor(images)
            features_flatten = features.reshape(features.size(0), -1)

            final_features = self.linear_to_embed_size(features_flatten)
            final_features = self.bn(final_features)

        elif(self.mode == "depthwise"):
            images_A = images[:, :3, :, :]
            images_B = images[:, 3:, :, :]
            
            with torch.no_grad():
                features_A = self.feature_extractor(images_A)
            features_A_flatten = features_A.reshape(features_A.size(0), -1)

            with torch.no_grad():
                features_B = self.feature_extractor(images_B)
            features_B_flatten = features_B.reshape(features_B.size(0), -1)

            final_features = self.linear_to_embed_size_concat(torch.cat((features_A_flatten, features_B_flatten), 1))
            #final_features = self.bn(final_features)

        return final_features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=50):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        print(embeddings.shape)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        print(packed)
        print(packed[0].shape)
        
        hiddens, _ = self.lstm(packed)

        outputs = self.linear(hiddens[0])
        print(outputs.shape)

        sys.exit()
        
        return outputs
    
    '''def sample(self, features, states=None, vocab = None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        must_have_words = ["<start>", "<end>"]
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, highest_predicted = outputs.max(1)                        # predicted: (batch_size)
            _, three_highest_predicted = torch.topk(outputs, 3)
            
            word = vocab.idx2word[int(three_highest_predicted.squeeze(0)[0].detach().cpu().numpy())]

            sampled_ids.append(three_highest_predicted.squeeze(0)[0].unsqueeze(0))
            inputs = self.embed(three_highest_predicted.squeeze(0)[0].unsqueeze(0))                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
            
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids'''

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            if(i <= 2):
                print(predicted)
            if(i == 2): sys.exit()
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids