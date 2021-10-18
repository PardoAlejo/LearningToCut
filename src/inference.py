from __future__ import division, print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import time
import os
import numpy as np
import random
import pandas as pd
from dataset import ShotsDataset
from model import CutsModel, CutsLoss
from config import Config, backward_compatible_config, dump_config_details_to_tensorboard

def create_datasets(config, args):
    datasets, data_loaders = {}, {}
    if 'train' in args.split:
        datasets['train_eval'] = ShotsDataset(features_file_names=args.features_file_names, 
                                    labels_filename=args.train_labels_filename,
                                    durations_filename=args.durations_filename, 
                                    mode='test',
                                    snippet_size=config['snippet_size'], 
                                    stride=config['stride'], offset=config['offset'], 
                                    include_end_time=config['include_end_time'], 
                                    include_start_time=config['include_start_time'], 
                                    include_context=config['include_context'],
                                    boundary_oracle=config['boundary_oracle'])

        data_loaders['train_eval'] = DataLoader(dataset=datasets['train_eval'], batch_size=config['batch_size'], 
                                    pin_memory=False, num_workers=config['num_workers'],
                                    shuffle=True, collate_fn=datasets['train_eval'].collate_fn)
        config['input_size'] = datasets['val'].feature_size

    if 'val' in args.split:
        datasets['val'] = ShotsDataset(features_file_names=args.features_file_names, 
                                    labels_filename=args.val_labels_filename,
                                    durations_filename=args.durations_filename,
                                    mode='test',
                                    snippet_size=config['snippet_size'], 
                                    stride=config['stride'], offset=config['offset'], 
                                    include_end_time=config['include_end_time'], 
                                    include_start_time=config['include_start_time'], 
                                    include_context=config['include_context'],
                                    boundary_oracle=config['boundary_oracle'])

        

        data_loaders['val'] = DataLoader(dataset=datasets['val'], batch_size=config['batch_size'], 
                                        pin_memory=False, num_workers=config['num_workers'],
                                        shuffle=True, collate_fn=datasets['val'].collate_fn)
        
        config['input_size'] = datasets['val'].feature_size

    return config, datasets, data_loaders

def process_one_batch(config, model, loss, features, targets, device, data_loader):
    logits, transformed_features = model(features.to(device))
    data_loader.dataset.save_predictions(targets, logits, transformed_features)
    loss_results = loss.compute_loss(logits, transformed_features, targets, device)

    metrics = {k: v.item() for (k,v) in loss_results.items()}

    return metrics

def full_epoch(config, model, loss, data_loader, th_distances, device):

    model.eval()
    num_snippets_per_shot = data_loader.dataset.num_features / data_loader.dataset.total_shots
    top_k = int(config['top_k']*num_snippets_per_shot/100) if config['top_k'] > 0 else config['top_k']
    
    accumulated_metrics = {'Method':'LTC'}
    with torch.no_grad():
        for features, targets in data_loader:
            metrics = process_one_batch(config=config, model=model, loss=loss, features=features, targets=targets, 
                                        device=device, data_loader=data_loader)

        for th_distance in th_distances:
            logging.info(f"Computing Model Metrics for distance {th_distance}")
            eval_results = data_loader.dataset.eval_saved_predictions(top_k, th_dist=th_distance)
            accumulated_metrics.update(eval_results)
        
        logging.info(f'Model Performance: {[(k, np.round(v*100,4)) for (k,v) in accumulated_metrics.items() if not isinstance(v,str)]}]')
    return accumulated_metrics, data_loader.dataset.snippet_predictions

def results_to_csv(metrics, csv_path):

    mode = 'a' if os.path.exists(csv_path) else 'w'
    df = pd.DataFrame(metrics, index=[0])
    with open(csv_path, mode) as f:
        df.to_csv(f, index=False, header=f.tell()==0)


def test(args, config, model, loss, dataloader, device, csv_dict, th_distances=[1]):

    model.to(device)

    all_random_results = {'Method':'Random'}
    all_raw_results = {'Method':'Raw'}
    for th_dist in th_distances:
        logging.info(f"Computing Random and Raw metrics for distance {th_dist}")
        random_results = dataloader.dataset.get_random_metrics(th_dist=th_dist)
        raw_results = dataloader.dataset.eval_saved_predictions(top_k_candidates=-1, th_dist=th_dist)
        all_random_results.update(random_results)
        all_raw_results.update(raw_results)
    
    csv_dict.update(all_random_results)
    results_to_csv(csv_dict, args.csv_path)
    
    csv_dict.update(all_raw_results)
    results_to_csv(csv_dict, args.csv_path)

    metrics, _ = full_epoch(config, model, loss, dataloader, th_distances, device)
    csv_dict.update(metrics)
    results_to_csv(csv_dict, args.csv_path)

def main(args):
    start_time = time.time()
    device = torch.device(args.device)
    logging.info(f"Computing inference from {args.log_dir}")
    logging.info(f'using device {args.device}')
    
    checkpoint = torch.load(f"{args.log_dir}")
    config = checkpoint['config']
    config = backward_compatible_config(config)

    logging.info(f'Seeding with seed {config["seed"]}')
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    logging.info(f'config {config}')

    config, datasets, data_loaders = create_datasets(config, args)
    config['top_k'] = args.top_k

    num_classes = 1 # Cut or not
    model = CutsModel(num_classes=num_classes, input_size=config['input_size'], num_layers=config['num_layers'])
    model.load_state_dict(checkpoint['model'], strict=False)
    
    loss = CutsLoss(contrastive_loss_type=config['contrastive_loss_type'], 
                    alpha_contrastive=config['alpha_contrastive'],
                    alpha_ce=config['alpha_ce'],
                    alpha_ce_pairs=config['alpha_ce_pairs'],
                    bce_pairs_logits=config['bce_pairs_logits'],
                    boundary_oracle=config['boundary_oracle'])    
    

    for key,dataloader in data_loaders.items():
        csv_dict = {}
        logging.info(f' ========== Inference on {key} split ===========')
        test(args, config, model, loss, dataloader, device, csv_dict, th_distances=args.th_distances)
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Training a Mantis classification model',
                        formatter_class=ArgumentDefaultsHelpFormatter)
    # required arguments.
    parser.add_argument('--features_file_names', required=True, type=str, action='append',
                        help='Path to an HDF5 files containing the train video features')
    parser.add_argument('--train_labels_filename', required=True, type=str,
                        help='Path to CSV file with the train ground truth labels')
    parser.add_argument('--val_labels_filename', required=True, type=str,
                        help='Path to CSV file with the test ground truth labels')
    parser.add_argument('--durations_filename', required=True, type=str,
                        help='Path to CSV file with the duration of the videos')
    parser.add_argument('--log_dir', required=True, type=str,
                        help='Where logs and model checkpoints are saved.')
    parser.add_argument('--top_k', default=-1, type=int,
                        help='Top k scoring snippets for ranking, -1 no top')
    parser.add_argument('--th_distances', default=1, type=int, nargs='+',
                        help='Distances to consider TP on evaluation') 
    parser.add_argument('--csv_path', default='./results.csv', type=str,
                        help='Csv path to save the results.')
    # optional arguments                   
    parser.add_argument('--batch_size', default=512, type=int,
                      help='Batch Size')
    parser.add_argument('--device', default='cuda:0', type=str,
                      help='The GPU device to train on')
    parser.add_argument('--config_type', default='basic', type=str,
                      help='The hyperparameter config type')
    parser.add_argument('--loglevel', default='INFO', type=str, help='logging level')
    parser.add_argument('--split', nargs='+', default=['val'], type=str,
                        choices=['val', 'train'], 
                        help='Datset split to run the inference on, train/val or both')               

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s',
                    level=numeric_level)
    delattr(args, 'loglevel')
    if not os.path.exists(os.path.dirname(args.csv_path)):
        os.makedirs(os.path.dirname(args.csv_path))
    main(args)