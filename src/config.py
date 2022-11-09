import numpy as np
import logging


class Config(object):
    def __init__(self, config_type):
        switcher = {'basic': self._create_basic_config,
                    'random': self._create_random_config,
                    'audio_video':self._best_audio_video_config}
        self.create_config = switcher.get(config_type, None)

        logging.info('Setting config type to {}'.format(config_type))
        if self.create_config is None:
            logging.warn('Invalid config type {}. Setting config type is to the default playground config.'.format(config_type))
            self.create_config = self._create_basic_config

    def _create_basic_config(self, batch_size, context, alpha_contrastive, alpha_ce, 
                            alpha_ce_pairs, bce_pairs_logits, contrastive_loss_type, 
                            top_k,boundary_oracle):
        logging.info('Generating a basic config')
        config = {'num_workers': 4,
                  'batch_size': batch_size,
                  'snippet_size': 16,
                  'stride': 8,
                  'offset': 1,
                  'num_layers': 2,
                  
                  'include_end_time': 1,
                  'include_start_time': 1,
                  'neg_pos_ratio': 1,
                  'include_context': context,
                  'boundary_oracle': boundary_oracle,
                  'top_k': top_k,
                  
                  'contrastive_loss_type': contrastive_loss_type,
                  'alpha_contrastive': alpha_contrastive,
                  'alpha_ce': alpha_ce,
                  'bce_pairs_logits': bce_pairs_logits,
                  'alpha_ce_pairs': alpha_ce_pairs,

                  'initial_lr': 10**-4,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'seed': 183141,
                  'version_name': 'basic_config',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']}
        return config
    
    def _best_audio_video_config(self, batch_size, context, alpha_contrastive, alpha_ce, 
                            alpha_ce_pairs, bce_pairs_logits, contrastive_loss_type, 
                            top_k,boundary_oracle):
        logging.info('Generating a basic config')
        config = {'num_workers': 4,
                  'batch_size': 128,
                  'snippet_size': 16,
                  'stride': 8,
                  'offset': 1,
                  'num_layers': 2,
                  
                  'include_end_time': 1,
                  'include_start_time': 1,
                  'neg_pos_ratio': 1,
                  'include_context': 0,
                  'boundary_oracle': 0,
                  'top_k': top_k,
                  
                  'contrastive_loss_type': 'nce',
                  'alpha_contrastive': 1.0,
                  'alpha_ce': 0.0,
                  'bce_pairs_logits': 0,
                  'alpha_ce_pairs': 0.0,

                  'initial_lr': 10**-3,
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'seed': 381037,
                  'version_name': 'basic_config',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']}
        return config
    
    def _best_audio_video_config(self, batch_size, context, alpha_contrastive, alpha_ce, 
                            alpha_ce_pairs, bce_pairs_logits, contrastive_loss_type, 
                            top_k,boundary_oracle):
        logging.info('Generating a basic config')
        config = {'num_workers': 4,
                  'batch_size': batch_size,
                  'snippet_size': 16,
                  'stride': 8,
                  'offset': 1,
                  'num_layers': rnd_choice(2,4,1,output_type=int),
                  
                  'include_end_time': 1,
                  'include_start_time': 1,
                  'neg_pos_ratio': 1,
                  'include_context': context,
                  'boundary_oracle': boundary_oracle,
                  'top_k': top_k,
                  
                  'contrastive_loss_type': contrastive_loss_type,
                  'alpha_contrastive': alpha_contrastive,
                  'alpha_ce': alpha_ce,
                  'bce_pairs_logits': bce_pairs_logits,
                  'alpha_ce_pairs': alpha_ce_pairs,

                  'initial_lr': 10**rnd_choice(-4,-3,-1,output_type=float),
                  'lr_decay': 0.9,
                  'lr_patience': 1,

                  'max_epoch': 50,
                  'seed': np.random.randint(424242),
                  'version_name': 'basic_config',
                  'do_not_dump_in_tensorboard': ['do_not_dump_in_tensorboard', 'version_name', 'tiou_thresholds']}
        return config

def rnd_choice(start, end, step=1, output_type=float):
    '''
    generates a random number in [start, end] with spacing 
    size equal to step. The value of end is included.
    '''
    nums = np.append(np.arange(start, end, step), end)
    return output_type(np.random.choice(nums))

def backward_compatible_config(config):
    return config

def dump_config_details_to_tensorboard(summary_writer, config):
    for k, v in config.items():
        if k not in config['do_not_dump_in_tensorboard']:
            if type(v) == str:
                continue
            summary_writer.add_scalar('config/{}'.format(k), v, 0)
