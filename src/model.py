import logging
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class CutsModel(nn.Module):

    def __init__(self, num_classes, input_size, num_layers):
        '''
        Args:
            num_classes (int): The number of labels to classify
            input_size (int): The size of each feature
            
        '''
        super().__init__()
        logging.info(f'Model: num_classes {num_classes} input_size {input_size} num_layers {num_layers} ')

        self.num_classes = num_classes
        self.input_size = input_size
        self.num_layers = num_layers
        self.feat_transform, mid_feat_size = self._create_sequential_linear_relu_layers(self.num_layers, self.input_size)
        self.relu = nn.ReLU()
        self.fc_class = nn.Linear(mid_feat_size, self.num_classes)

    @staticmethod
    def _create_sequential_linear_relu_layers(num_layers, input_size, output_size=None, reduction_factor=2):
        layers = OrderedDict()
        this_input_size = input_size
        for i in range(num_layers-1):
            if output_size:
                this_output_size = max(int(this_input_size/reduction_factor), output_size)
            else:
                this_output_size = int(this_input_size/reduction_factor)
            # this_output_size = max(int(this_input_size/2), output_size)
            layers[f'linear_{i}'] = nn.Linear(this_input_size, this_output_size)
            layers[f'relu_{i}'] = nn.ReLU()
            this_input_size = this_output_size
        if output_size:
            layers[f'linear_{num_layers-1}'] = nn.Linear(this_input_size, output_size)
            this_input_size = output_size
        else:
            del layers[f'relu_{i}']

        return nn.Sequential(layers), this_input_size

    def forward(self, features):
        transformed_features = self.feat_transform(features)
        relu_features = self.relu(transformed_features)
        logits = self.fc_class(relu_features)
        return logits, transformed_features


class CutsLoss():
    def __init__(self, contrastive_loss_type, alpha_contrastive, alpha_ce_pairs, bce_pairs_logits=False, alpha_ce=0, boundary_oracle=False):
        '''
        Args:
            Contrastive loss type (str): type of contrastive loss
            alpha_contrastive (float): Weight for the loss
            bce_pairs_logits compute the bce from logits not from features
        '''

        self.alpha_contrastive = alpha_contrastive
        self.alpha_ce = alpha_ce
        self.alpha_ce_pairs = alpha_ce_pairs
        self.bce_pairs_logits = bce_pairs_logits
        self.contrastive_loss_type = contrastive_loss_type
        self.boundary_oracle = boundary_oracle
        
        if self.bce_pairs_logits:
            self.beta_pairs = 0.5
        else:
            self.beta_pairs = 0.005

        if self.contrastive_loss_type == 'triplet':
            self.contrastive_loss = self.triplet_loss
            self.beta_contrastive = 1
        elif self.contrastive_loss_type == 'nce':
            self.contrastive_loss = self.nce_loss
            self.beta_contrastive = 0.2
        else:
            raise IOError(f'Invalid Loss type. Error message: {e}')

    def compute_loss(self, logits, features, targets, device):

        loss_results = {'cross_entropy_loss': torch.tensor(0.0).to(device),
                        'constrastive_loss': torch.tensor(0.0).to(device),
                        'pairs_class_loss': torch.tensor(0.0).to(device),
                        'loss': torch.tensor(0.0).to(device),
                        }
        num_valid_videos = 0
        for i, video_name in enumerate(targets['video-names']):
            this_pairs = targets['video-name-to-pairs'][video_name]
            if len(this_pairs) == 0:
                continue
            num_valid_videos =+ 1
            start, end = targets['video-name-to-slice'][video_name]
            this_logits = logits[start:end].squeeze(-1)
            this_features = features[start:end]
            this_labels = targets['labels'][start:end]
            this_masks_shot_idx = targets['masks_shot_idx'][start:end]
            backprop_idx = this_labels >= 0
            this_loss_results = self._loss_for_one_video(logits=this_logits,
                                                        features=this_features,
                                                        labels=this_labels,
                                                        backprop_idx=backprop_idx,
                                                        masks_shot_idx=this_masks_shot_idx,
                                                        pairs=this_pairs,
                                                        snippet_weight=targets['snippet_class_weight'],
                                                        pairs_weight=targets['pairs_class_weight'],
                                                        device=device)
            for k, v in this_loss_results.items():
                loss_results[k] += v

        for k, v in loss_results.items():
            loss_results[k] /= len(targets['video-names'])

        return loss_results

    def _loss_for_one_video(self, logits, features, labels, backprop_idx, masks_shot_idx, pairs, snippet_weight, pairs_weight, device):
        '''
            TODO: Specify here the dimensions
            cas: 1 x T x num_classes
            attention" 1 x T x {1,2}
            labels: 1 x 1
            pseudo_gt: 1 x T
            bg_cas: 1 x T X num_classes+1
        '''
        snippet_class_loss = F.binary_cross_entropy_with_logits(input=logits[backprop_idx].view(1, -1),
                                                    target=labels[backprop_idx].unsqueeze(0).type(torch.cuda.FloatTensor).to(device),
                                                    pos_weight=torch.tensor(snippet_weight).to(device))
        
        if self.bce_pairs_logits:
            scores = torch.matmul(logits.unsqueeze(1),logits.unsqueeze(0))/2
        else:
            scores = torch.matmul(features, features.transpose(0,1))
        
        pairs_class_loss = F.binary_cross_entropy_with_logits(input=scores[pairs[:, 0], pairs[:, 1]].to(device),
                                                            target=pairs[:,2].type(torch.cuda.FloatTensor).to(device),
                                                            pos_weight=torch.tensor(pairs_weight).to(device))

        sim_loss = 0
        num_valid_shots = 0
        # Find triplet loss per shot
        for shot_idx in range(1,max(masks_shot_idx)):
            this_idx_shot = torch.where(masks_shot_idx == shot_idx)[0]
            next_idx_shot = torch.where(masks_shot_idx == shot_idx +1)[0]
            # Check if the two shots are adjacent or not, they have to have two zeros between them
            if next_idx_shot[0] - this_idx_shot[-1] != 3:
                continue
            this_features = features[masks_shot_idx == shot_idx]
            neighbour_features = features[masks_shot_idx == shot_idx+1]
            if neighbour_features.shape[0]<=1:
                continue
            num_valid_shots += 1
            positive_pair = (this_features[-1], neighbour_features[0])
            
            if self.boundary_oracle:
                anchor_features = this_features[-1:]
            else:
                anchor_features = this_features

            sim_loss += self.contrastive_loss(positive_pair, anchor_features, neighbour_features[1:])

        contrastive_loss = sim_loss/max(1,num_valid_shots) 

        final_contrastive_loss = self.alpha_contrastive * self.beta_contrastive * contrastive_loss
        final_shot_class_loss = self.alpha_ce * snippet_class_loss
        final_pair_class_loss = self.alpha_ce_pairs* self.beta_pairs * pairs_class_loss

        loss = final_contrastive_loss + final_shot_class_loss + final_pair_class_loss

        loss_results = {'cross_entropy_loss': snippet_class_loss,
                        'constrastive_loss': contrastive_loss,
                        'pairs_class_loss': pairs_class_loss,
                        'loss': loss}

        return loss_results

    def triplet_loss(self, positive_pair, anchor_features, negative_features, triplet_margin=1):
        pos_distance = torch.norm(positive_pair[0] - positive_pair[1], dim=-1)
        neg_distances = torch.norm(anchor_features.unsqueeze(0) - negative_features.unsqueeze(1), dim=-1).view(-1)    
        triplet_loss = nn.functional.relu((pos_distance - neg_distances) + triplet_margin).mean()
        return triplet_loss

    def nce_loss(self, positive_pair, anchor_features, negative_features):
        positive_1 = positive_pair[0]/(torch.norm(positive_pair[0], dim=-1, keepdim=True))
        positive_2 = positive_pair[1]/(torch.norm(positive_pair[1], dim=-1, keepdim=True))
        anchor_features_norm = anchor_features/(torch.norm(anchor_features, dim=-1, keepdim=True))
        negative_features_norm = negative_features/(torch.norm(negative_features, dim=-1, keepdim=True))
        pos_score = torch.matmul(positive_1, positive_2).exp()
        neg_score = torch.matmul(anchor_features_norm, negative_features_norm.transpose(0,1)).exp().sum()
        sim_loss = -1*((pos_score)/(pos_score+neg_score)).log()
        return sim_loss