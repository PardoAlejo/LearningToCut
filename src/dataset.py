import logging
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py
from sklearn.metrics import accuracy_score, average_precision_score
from scipy.special import softmax
from skimage.measure import label as label_connected_component
import time
import json
import random 
import operator 

class ShotsDataset(Dataset):
    """Construct an untrimmed video classification dataset."""

    def __init__(self,
                 features_file_names,
                 labels_filename,
                 durations_filename,
                 mode='test',
                 neg_pos_ratio=None,
                 snippet_size=16,
                 stride = 8,
                 fps=24,
                 offset = 1,
                 include_end_time = True,
                 include_start_time = False,
                 include_context = True,
                 boundary_oracle = False,
                 seed=424242):
        """
        Parameters
        ----------
        features_filename : str.
            Path to a features HDF5 file.
        labels_filename : str.
            Path to a CSV file. columns: video-name,t-start,t-end,label,video-duration,fps,video-num-frames
        snippet_size : int.
            Number of frames per snippet
        mode: train or test (inference)
        """
        assert(features_file_names), "At least one feature file needed" 
        # logging.info(f'Dataset: features_filename {features_filename} labels_filename {labels_filename} snippet_size {snippet_size}'
        #             f' stride {stride}')
        np.random.seed(seed)
        self.include_context = include_context
        self.snippet_size = snippet_size
        self.fps = fps
        self.offset = offset
        self.stride = stride
        self.num_features = 0
        self.num_positive = 0
        self.num_negative = 0
        self.num_pairs = None
        self.num_positive_pairs = None
        self.total_shots = None
        self.total_valid_shots = None
        self.snippet_pos_weight = None
        self.pairs_pos_weight = None
        self._video_names = 0
        self.boundary_oracle = boundary_oracle

        self._video_name_to_snippets_shots_gt = {}
        self._video_name_to_metadata = {}
        self._video_name_to_shot_idx = {}
        self._video_name_to_snippets_features = {}
        self._video_name_to_pairs = {}
        self._video_name_to_included_snippets = {}

        self.neg_pos_ratio = neg_pos_ratio
        self.feature_size = 0
        self.dtype = None
        self.features_dimensions = []

        self.subset = 'train' if 'train' in labels_filename else 'val'
        self.mode = mode
        self.debug = True if 'debug' in labels_filename else False

        self._set_durations(durations_filename)
        self._set_video_names(labels_filename)
        self._set_metadata()

        self.include_end_time = include_end_time
        self.include_start_time = include_start_time

        self._set_labels()

        self._set_pairs()

        # Set dimensions
        for i, feature_file_name in enumerate(features_file_names):
            self.set_feature_dimensions(feature_file_name)
        # Set features
        for i, feature_file_name in enumerate(features_file_names):
            logging.info(f"Setting features from {feature_file_name}")
            self._set_features(feature_file_name, file_number=i)

        if self.include_context:
            self._set_context()
        
        logging.info(f"Dimension of features: {self.feature_size}")

        self.snippet_predictions = dict(zip(self._video_names, len(self._video_names)*[None]))
        self.video_name_to_similarity = {}
        self.initialize_similarities()

    def __len__(self):
        return len(self._video_names)

    def __getitem__(self, idx):
        
        video_name = self._video_names[idx]
        included = self._video_name_to_included_snippets[video_name]
        features = torch.tensor(self._video_name_to_snippets_features[video_name][included], dtype=torch.float)
        label = torch.tensor(self._video_name_to_snippets_shots_gt[video_name], dtype=torch.long)
        mask_shot_idx = torch.tensor(self._video_name_to_shot_idx[video_name], dtype=torch.long)
        pairs = torch.tensor([pair for shot in [v for k,v in self._video_name_to_pairs[video_name].items()] for pair in shot], dtype=torch.long)
        return video_name, features, label, mask_shot_idx, pairs

    def collate_fn(self, data_lst):
        video_names = [_video_name for _video_name, _, _, _, _ in data_lst]
        video_name_to_slice = {}
        _video_name_to_pairs = {}
        current_ind = 0
        for this_video_name, this_features, _, _, pairs in data_lst:
            video_name_to_slice[this_video_name] = (current_ind, current_ind + this_features.shape[0])
            _video_name_to_pairs[this_video_name] = pairs
            current_ind += this_features.shape[0]
        features    = torch.cat([_features for _, _features, _, _, _ in data_lst],dim=0)
        labels      = torch.cat([_label for _, _, _label, _, _ in data_lst],dim=0)
        masks_shot_idx       = torch.cat([_mask for _, _, _, _mask, _ in data_lst],dim=0)

        targets = {'video-names': video_names,
                   'video-name-to-slice': video_name_to_slice,
                   'labels': labels,
                   'masks_shot_idx': masks_shot_idx,
                   'video-name-to-pairs': _video_name_to_pairs,
                   'snippet_class_weight': self.snippet_pos_weight,
                   'pairs_class_weight': self.pairs_pos_weight 
                   }
        return features, targets

    def _set_video_names(self, labels_filename):

        try:
            df = pd.read_csv(labels_filename)
            self.df = df
        except Exception as e:
            raise IOError(f'Invalid labels_filename. Error message: {e}')
    
        self._video_names = list(set(self.df['video_id']))
        logging.info(f"{len(self._video_names)} videos for {self.subset}")

    def _set_durations(self, durations_filename):
            
        try:
            df = pd.read_csv(durations_filename)
            self.durations_df = df
        except Exception as e:
            raise IOError(f'Invalid durations filename. Error message: {e}')

    def set_feature_dimensions(self, feature_path):
        try:
            features_dict = h5py.File(feature_path, 'r')
        except Exception as e:
            raise IOError(f'Invalid HDF5. Error message: {e}')
        feat_dim = features_dict[list(features_dict.keys())[0]].shape[-1]
        self.features_dimensions.append(feat_dim)
        features_dict.close()

    def _set_metadata(self):

        for video in self._video_names:
            # Save metadata
            this_duration = self.durations_df[self.durations_df.videoid==video].duration.values[0]
            this_num_features = int(np.ceil(this_duration*self.fps/self.stride))
            self.num_features += this_num_features
            self._video_name_to_metadata[video] = {'duration': this_duration,
                                                'fps': self.fps, 'num_frames': int(this_num_features * self.stride), 
                                                'num_features': this_num_features}

    def _set_labels(self):
        
        self.num_positive = 0
        self.num_negative = 0
        df_by_video_name = self.df.groupby(by='video_id')
        logging.info('Setting Labels')
        for video_name, this_df in df_by_video_name:

            this_num_features = self._video_name_to_metadata[video_name]['num_features']
            video_duration = self._video_name_to_metadata[video_name]['duration']
            
            bg_fg = np.zeros(int(np.ceil(this_num_features)), dtype=np.int)
            short_shots = 0
            total_shots = 0
            for _, row in this_df.iterrows():
                duration = row['end_time']-row['start_time']
                # Remove low confidence
                if row['confidence'] > 0.95 and (row['end_time'] < video_duration):
                    # Remove short shots with less than 24 frames (3 features)
                    total_shots += 1
                    if duration <= 24/self.fps:
                        short_shots += 1
                        frame_shot_start = int(np.floor(row['start_time'] * self.fps))
                        frame_shot_end = int(np.floor(row['end_time'] * self.fps))
                        start_snippet_idx = int(np.floor(frame_shot_start/self.stride))
                        end_snippet_idx = int(np.floor(frame_shot_end/self.stride))
                        bg_fg[max(0,start_snippet_idx - 1):min(this_num_features, end_snippet_idx + 1)] = -1
                        continue

                    if self.include_start_time:
                        frame_shot_start = int(np.floor(row['start_time'] * self.fps))
                        snippet_idx = int(np.floor(frame_shot_start/self.stride))
                        # Find the previous neighboor that doesn't contain the transition
                        # For start the one next to snippet_idx
                        neighboor_idx = min(this_num_features, snippet_idx + (self.offset))
                        bg_fg[neighboor_idx] = 1
                        # Remove the snippets with the shot transition
                        bg_fg[max(0,snippet_idx-1):min(this_num_features, snippet_idx+1)] = -1

                    if self.include_end_time:
                        frame_shot_end = int(np.floor(row['end_time'] * self.fps))
                        snippet_idx = int(np.floor(frame_shot_end/self.stride))
                        # Find the previous neighboor that doesn't contain the transition
                        # For end 2 snippets before snippet idx due to overlap
                        neighboor_idx = max(0, snippet_idx - (self.offset + 1))
                        bg_fg[neighboor_idx] = 1
                        # Remove the snippets with the shot transition
                        bg_fg[max(0,snippet_idx-1):min(this_num_features, snippet_idx+1)] = -1
                
                else:
                    continue
            
            this_num_positives = np.count_nonzero(bg_fg[bg_fg==1])
            if this_num_positives == 0:
                self._video_names.remove(video_name)
                continue

            if self.mode == 'train':
                # Sample few negatives for training based on neg_pos_ratio
                included_snippets = np.zeros(bg_fg.shape, dtype=bool)
                idxs_negs = random.choices(np.where(bg_fg == 0)[0],k=this_num_positives*self.neg_pos_ratio) 
                idxs_pos = np.where(bg_fg == 1)[0]
                idxs_ign = np.where(bg_fg == -1)[0]
                included_snippets[np.concatenate((idxs_negs,idxs_pos,idxs_ign),axis=0)] = True
                self._video_name_to_included_snippets[video_name] = included_snippets
            else:
                # Include all for validation
                included_snippets = np.where(bg_fg >= -1)
                self._video_name_to_included_snippets[video_name] = included_snippets

            labeled_mask = label_connected_component(bg_fg[included_snippets]>=0, connectivity=1)
            
            self._video_name_to_shot_idx[video_name] = labeled_mask
            self.num_positive += this_num_positives
            self.num_negative += np.count_nonzero(bg_fg[included_snippets][bg_fg[included_snippets] == 0])
            self._video_name_to_snippets_shots_gt[video_name] = bg_fg[included_snippets]

        self.snippet_pos_weight = self.num_negative/self.num_positive
        logging.info(f"{short_shots*100/total_shots:.2f} % of ignored shots")
        logging.info(f"{self.num_positive*100/self.num_features:.1f}% positive instances among {self.num_features} in {self.subset}")

    def _set_pairs(self):

        logging.info("Computing pairs")
        self.total_shots = 0
        self.total_valid_shots = 0
        self.num_pairs = 0
        self.num_positive_pairs = 0
        tick = time.time()
        for video_name in self._video_names:
            this_mask_shot_idx = self._video_name_to_shot_idx[video_name]
            self._video_name_to_pairs[video_name] = {}
            for shot in range(1,max(this_mask_shot_idx)):
                self.total_shots += 1
                this_idx_shot = np.where(this_mask_shot_idx == shot)[0]
                next_idx_shot = np.where(this_mask_shot_idx == shot +1)[0]
                if self.boundary_oracle:
                    query_shot = this_idx_shot[-1:]
                else:
                    query_shot = this_idx_shot
                # distance between two adjancent border snippets is 3
                this_pairs = [( x, y, 1 if y-x==3 else 0) for x in query_shot for y in next_idx_shot]
                # Only include shots with positive examples
                if any(x[-1]==1 for x in this_pairs):
                    self.num_pairs += len(this_pairs)
                    self.num_positive_pairs += np.sum([x[2] for x in this_pairs])
                    self.total_valid_shots += 1
                    self._video_name_to_pairs[video_name].update({shot:this_pairs})

        tock = time.time()
        logging.info(f'{tock-tick:.3f} seconds calculating pairs for {self.subset}')
        
        self.pairs_pos_weight = (self.num_pairs - self.num_positive_pairs)/self.num_positive_pairs

        logging.info(f"Total number of shots for {self.subset}: {self.total_shots}")         
        logging.info(f"Included shots for {self.subset}: {self.total_valid_shots}, {self.total_valid_shots*100/self.total_shots:.2f}% of total")
        logging.info(f"Total number of pairs for {self.subset}: {self.num_pairs}")
        logging.info(f"{self.num_positive_pairs*100/self.num_pairs:.3f}% of positive pairs in {self.subset}")
        logging.info(f"Around {self.num_features/self.total_shots:.0f} snippets per shot in {self.subset}")
        logging.info(f"Around {self.num_pairs/self.total_shots:.0f} pairs per shot in {self.subset}")

    def set_random_features(self, feature_path, file_number=0):
        for video in self._video_names:
            this_num_features = self._video_name_to_metadata[video]['num_features']
            feat_dim = self.features_dimensions[file_number]
            self._video_name_to_snippets_features[video] = np.random.rand(this_num_features, feat_dim)
        temp = self._video_name_to_snippets_features[video]
        self.feature_size = temp.shape[-1]
        self.dtype = temp.dtype

    def _set_features(self, feature_path, file_number=0):

        try:
            features_dict = h5py.File(feature_path, 'r')
        except Exception as e:
            raise IOError(f'Invalid HDF5 for the visual observations. Error message: {e}')

        for video in self._video_names:
            # Save metadata
            this_num_features = self._video_name_to_metadata[video]['num_features']

            feat_dim = np.sum(self.features_dimensions)
            feat_dim = feat_dim*2 if self.include_context else feat_dim
            
            # Place features
            if file_number == 0:
                self._video_name_to_snippets_features[video] = np.zeros([this_num_features, feat_dim], dtype=np.float16)

            if video in features_dict.keys():
                feat_start_idx = np.sum(self.features_dimensions[:file_number], dtype=np.int)
                feat_end_idx = np.sum(self.features_dimensions[:file_number+1], dtype=np.int)
                self._video_name_to_snippets_features[video][:,feat_start_idx:feat_end_idx] = features_dict[video][()][:this_num_features]
            else:
                continue

        features_dict.close()
        temp = self._video_name_to_snippets_features[video]
        self.feature_size = temp.shape[-1]
        self.dtype = temp.dtype

    def _set_context(self):

        for video_name in self._video_names:
            this_labeled_mask = self._video_name_to_shot_idx[video_name]
            this_num_shots = max(this_labeled_mask)
            this_video_features = self._video_name_to_snippets_features[video_name]
            for shot_number in range(1,this_num_shots+1):
                this_shot_idxs = np.where(this_labeled_mask==shot_number)[0]
                # Mean pool features indicated by idxs
                feat_size = int(self.feature_size/2)
                # feat_size = self.feature_size
                neighborhood_features = np.max(this_video_features[this_shot_idxs][:,:feat_size],axis=0)
                this_video_features[this_shot_idxs,feat_size:] = neighborhood_features[np.newaxis,:]
                # this_video_features[this_shot_idxs] += neighboor_features[np.newaxis,:]

    def initialize_similarities(self):
        logging.info("Initializing similarities")
        for video_name in self._video_names: 
            this_features = torch.from_numpy(self._video_name_to_snippets_features[video_name]).cuda()
            self.snippet_predictions[video_name] = np.random.rand(this_features.shape[0],1).astype(np.double)
            similarities = torch.matmul(this_features,this_features.transpose(0,1))
            self.video_name_to_similarity[video_name] = similarities.cpu().detach().numpy()

    def save_predictions(self, targets, logits, transformed_features):
        
        logits = logits.cpu().detach().numpy()
        transformed_features = transformed_features.detach()
        for i, video_name in enumerate(targets['video-names']):
            start, end = targets['video-name-to-slice'][video_name]
            self.snippet_predictions[video_name]  = logits[start:end].astype(np.double)
            this_features = transformed_features[start:end]
            similarities = torch.matmul(this_features,this_features.transpose(0,1))
            self.video_name_to_similarity[video_name] = similarities.cpu().numpy().astype(np.double)
    
    def eval_ranking_and_pairs_AP(self, th_dist):
        ranking_results = []
        all_pair_labels = []
        all_pair_scores = []
        dist_shots = 2
        for video_name, video_dict in self._video_name_to_pairs.items():
            video_similarities = self.video_name_to_similarity[video_name]
            for _, shot_pairs in video_dict.items():
                shot_labels = []
                shot_similarities = []
                for pair in shot_pairs:
                    # pair contains (query, candidate, label)
                    query = pair[0]
                    candidate = pair[1]
                    if th_dist == 1:
                        shot_labels.append(pair[2])
                    else:
                        shot_labels.append(1 if candidate-query <= (th_dist+dist_shots) else 0)
                    shot_similarities.append(video_similarities[query,candidate])
                all_pair_labels.extend(shot_labels)
                all_pair_scores.extend(shot_similarities)
        
        connected_components = list(label_connected_component(np.array(all_pair_labels)))
        _, connected_components_sorted = zip(*sorted(zip(all_pair_scores, connected_components), key=operator.itemgetter(0), reverse = True))

        results = {
        f'AR1-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=1),
        f'AR3-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=3),
        f'AR5-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=5),
        f'AR10-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=10)}

        results.update(self._eval_pairs_label_accuracy(all_pair_labels, all_pair_scores, th_dist=th_dist))
        
        return results
    

    def eval_top_k_ranking_and_pairs_AP(self, k, th_dist):
        logging.info(f"Computing ranking with top {k} snippet scores for {self.subset}")
        ranking_results = []
        all_pair_labels = []
        all_pair_scores = []
        dist_shots=2
        for video_name, video_dict in self._video_name_to_pairs.items():
            video_similarities = self.video_name_to_similarity[video_name]
            snippet_scores = self.snippet_predictions[video_name].squeeze()
            for _, shot_pairs in video_dict.items():
                shot_labels = []
                shot_similarities = []
                # Get all idx for both shots
                this_shot_idx = np.array(list(set([pair[0] for pair in shot_pairs])))
                next_shot_idx = np.array(list(set([pair[1] for pair in shot_pairs])))
                #Check top scoring snippets for both shots
                top_k_this_shot = this_shot_idx[np.argpartition(snippet_scores[this_shot_idx],-(min(this_shot_idx.shape[0],k)))[-k:]]
                top_k_next_shot = next_shot_idx[np.argpartition(snippet_scores[next_shot_idx],-(min(next_shot_idx.shape[0],k)))[-k:]]
                for pair in shot_pairs:
                    # pair contains (query, candidate, label)
                    query = pair[0]
                    candidate = pair[1]

                    # Include only top scoring snippets
                    if query in top_k_this_shot and candidate in top_k_next_shot:
                        shot_similarities.append(video_similarities[query,candidate])
                    else:
                        # Give a very low similarity to ignore them in the ranking
                        shot_similarities.append(0)
                    
                    if th_dist == 1:
                        shot_labels.append(pair[2])
                    else:
                        shot_labels.append(1 if candidate-query <= (th_dist+dist_shots) else 0)

                all_pair_labels.extend(shot_labels)
                all_pair_scores.extend(shot_similarities)

        connected_components = list(label_connected_component(np.array(all_pair_labels)))
        _, connected_components_sorted = zip(*sorted(zip(all_pair_scores, connected_components), key=operator.itemgetter(0), reverse = True))

        results = {
        f'AR1-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=1),
        f'AR3-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=3),
        f'AR5-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=5),
        f'AR10-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=10)}

        results.update(self._eval_pairs_label_accuracy(all_pair_labels, all_pair_scores, th_dist=th_dist))
        
        return results

    def eval_ranking_and_pairs_AP_random(self, th_dist):
        ranking_results = []
        all_pair_labels = []
        all_pair_scores = []
        num_pairs = []
        num_small_pairs = 0
        dist_shots = 2
        for video_name, video_dict in self._video_name_to_pairs.items():
            for _, shot_pairs in video_dict.items():
                shot_labels = []
                shot_similarities = []
                for pair in shot_pairs:
                    # pair contains (query, candidate, label)
                    query = pair[0]
                    candidate = pair[1]
                    if th_dist == 1:
                        shot_labels.append(pair[2])
                    else:
                        shot_labels.append(1 if candidate-query <= (th_dist+dist_shots) else 0)
                    shot_similarities.append(np.random.rand())
                    # shot_similarities.append(1)
                all_pair_labels.extend(shot_labels)
                all_pair_scores.extend(shot_similarities)
        
        connected_components = list(label_connected_component(np.array(all_pair_labels)))
        _, connected_components_sorted = zip(*sorted(zip(all_pair_scores, connected_components), key=operator.itemgetter(0), reverse = True))

        results = {
        f'AR1-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=1),
        f'AR3-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=3),
        f'AR5-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=5),
        f'AR10-D{th_dist}': ranking_metric(connected_components_sorted, num_cuts=self.total_valid_shots, top_k_recall=10)}

        results.update(self._eval_pairs_label_accuracy(all_pair_labels, all_pair_scores, th_dist=th_dist))

        return results

    def _eval_pairs_label_accuracy(self, pairs_labels, pairs_similarities, th_dist=1):
        soft_predictions = sigmoid(np.array(pairs_similarities))
        ap = average_precision_score(pairs_labels, soft_predictions)
        results = {f'ap_pairs-D{th_dist}': ap}
        return results

    def _eval_snippets_label_accuracy(self):
        labels, soft_predictions = [], []
        
        for video_name in self._video_names:
            this_logits = self.snippet_predictions[video_name]
            this_labels = self._video_name_to_snippets_shots_gt[video_name]
            eval_idx = this_labels >= 0
            labels.extend(this_labels[eval_idx].tolist())
            this_sigmoid_logits = sigmoid(this_logits[eval_idx].squeeze(axis=-1))
            soft_predictions.extend(this_sigmoid_logits.tolist())

        ap = average_precision_score(labels, soft_predictions)
        results = {'ap_snippets': ap}

        return results

    def eval_saved_predictions(self, top_k_candidates=-1, th_dist=1):
        results = {}
        tick = time.time()
        
        if top_k_candidates == -1:
            results.update(self.eval_ranking_and_pairs_AP(th_dist=th_dist))
        else:
            results.update(self.eval_top_k_ranking_and_pairs_AP(k=top_k_candidates, th_dist=th_dist))
        return results

    def get_random_metrics(self,th_dist=1):
        results = {}
        tick = time.time()
        results.update(self.eval_ranking_and_pairs_AP_random(th_dist=th_dist))
        return results

def sigmoid(X):
    return 1/(1+np.exp(-X))

def ranking_metric(connected_components_sorted, num_cuts=None, top_k_recall=None):

    top_scoring_pairs = connected_components_sorted[:top_k_recall*num_cuts]
    num_unique_retrieved_cuts = np.unique(top_scoring_pairs).size - 1 
    recall = num_unique_retrieved_cuts/num_cuts
    return recall
    
