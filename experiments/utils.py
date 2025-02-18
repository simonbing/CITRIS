"""
General training function with PyTorch Lightning
"""

import os
import argparse
import json

import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint  #, GPUStatsMonitor
from shutil import copyfile
from copy import copy

import sys
sys.path.append('../')
from crc.baselines.citris.experiments.datasets import InterventionalPongDataset, ChambersDataset, ChambersSemiSynthDataset
from crc.utils.chamber_sim.simulators.lt.image import DecoderSimple

def get_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--causal_encoder_checkpoint', type=str, required=True)
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--exclude_vars', type=str, nargs='+', default=None)
    parser.add_argument('--exclude_objects', type=int, nargs='+', default=None)
    parser.add_argument('--coarse_vars', action='store_true')
    parser.add_argument('--data_img_width', type=int, default=-1)
    parser.add_argument('--seq_len', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--imperfect_interventions', action='store_true')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=-1)
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default='')
    return parser


def load_datasets(seed, dataset_name, data_dir, seq_len, batch_size, num_workers, args=None,
                  exclude_objects=None):
    pl.seed_everything(seed)
    print('Loading datasets...')
    match dataset_name:
        case 'chambers':
            data_name = dataset_name
            full_dataset = ChambersDataset(dataset='lt_crl_benchmark_v1',
                                           data_root=data_dir)

            train_frac = 0.8
            val_frac = 0.1
            test_frac = 0.1

            all_indxs = np.arange(len(full_dataset))
            train_idxs = all_indxs[:int(len(all_indxs) * train_frac)]
            val_idxs = all_indxs[len(train_idxs):len(train_idxs) + int(
                len(all_indxs) * val_frac)]
            test_idxs = all_indxs[max(val_idxs) + 1:]

            train_dataset = data.Subset(full_dataset, train_idxs)
            val_dataset = data.Subset(full_dataset, val_idxs)
            test_dataset = data.Subset(full_dataset, test_idxs)

            val_dataset_indep = ChambersDataset(dataset='lt_crl_benchmark_v1',
                                                data_root=data_dir,
                                                single_image=True,
                                                return_latents=True,
                                                mode='val')
            test_dataset_indep = ChambersDataset(dataset='lt_crl_benchmark_v1',
                                                 data_root=data_dir,
                                                 single_image=True,
                                                 return_latents=True,
                                                 mode='test')


            val_triplet_dataset = None
            test_triplet_dataset = None
        case 'chambers_semi_synth_decoder':
            data_name = dataset_name
            # decoder_simu = DecoderSimple()
            # transform = decoder_simu.simulate_from_inputs
            # full_dataset = ChambersSemiSynthDataset(dataset='lt_crl_benchmark_v1',
            #                                         data_root=data_dir,
            #                                         transform=transform)
            full_dataset = ChambersDataset(dataset='lt_crl_benchmark_v1',
                                           data_root=data_dir,
                                           exp_name='_synth_det')
            train_frac = 0.8
            val_frac = 0.1
            test_frac = 0.1

            all_indxs = np.arange(len(full_dataset))
            train_idxs = all_indxs[:int(len(all_indxs) * train_frac)]
            val_idxs = all_indxs[len(train_idxs):len(train_idxs)+int(len(all_indxs) * val_frac)]
            test_idxs = all_indxs[max(val_idxs)+1:]

            train_dataset = data.Subset(full_dataset, train_idxs)
            val_dataset = data.Subset(full_dataset, val_idxs)
            test_dataset = data.Subset(full_dataset, test_idxs)

            val_dataset_indep = ChambersDataset(dataset='lt_crl_benchmark_v1',
                                                data_root=data_dir,
                                                single_image=True,
                                                return_latents=True,
                                                mode='val',
                                                exp_name='_synth_det')
            test_dataset_indep = ChambersDataset(dataset='lt_crl_benchmark_v1',
                                                 data_root=data_dir,
                                                 single_image=True,
                                                 return_latents=True,
                                                 mode='test',
                                                 exp_name='_synth_det')

            # val_dataset_indep = ChambersSemiSynthDataset(dataset='lt_camera_v1',
            #                                              data_root=data_dir,
            #                                              single_image=True,
            #                                              return_latents=True,
            #                                              mode='val',
            #                                              transform=transform)
            # test_dataset_indep = ChambersSemiSynthDataset(dataset='lt_camera_v1',
            #                                               data_root=data_dir,
            #                                               single_image=True,
            #                                               return_latents=True,
            #                                               mode='test',
            #                                               transform=transform)

            dataset_args = {}

            val_triplet_dataset = None
            test_triplet_dataset = None
        case _:
            if 'ball_in_boxes' in data_dir:
                data_name = 'ballinboxes'
                DataClass = BallInBoxesDataset
                dataset_args = {}
                test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
            elif 'pong' in data_dir:
                data_name = 'pong'
                DataClass = InterventionalPongDataset
                dataset_args = {}
                test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
            elif 'causal3d' in data_dir:
                data_name = 'causal3d'
                DataClass = Causal3DDataset
                dataset_args = {'coarse_vars': args.coarse_vars, 'exclude_vars': args.exclude_vars, 'exclude_objects': args.exclude_objects}
                test_args = lambda train_set: {'causal_vars': train_set.full_target_names}
            elif 'voronoi' in data_dir:
                extra_name = data_dir.split('voronoi')[-1]
                if extra_name[-1] == '/':
                    extra_name = extra_name[:-1]
                extra_name = extra_name.replace('/', '_')
                data_name = 'voronoi' + extra_name
                DataClass = VoronoiDataset
                dataset_args = {}
                test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
            elif 'pinball' in data_dir:
                data_name = 'pinball' + data_dir.split('pinball')[-1].replace('/','')
                DataClass = PinballDataset
                dataset_args = {}
                test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
            else:
                assert False, f'Unknown data class for {data_dir}'
            train_dataset = DataClass(
                data_folder=data_dir, split='train', single_image=False, triplet=False, seq_len=seq_len, **dataset_args)
            val_dataset = DataClass(
                data_folder=data_dir, split='val_indep', single_image=True, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset))
            val_triplet_dataset = DataClass(
                data_folder=data_dir, split='val', single_image=False, triplet=True, return_latents=True, **dataset_args, **test_args(train_dataset))
            test_dataset = DataClass(
                data_folder=data_dir, split='test_indep', single_image=True, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset))
            test_triplet_dataset = DataClass(
                data_folder=data_dir, split='test', single_image=False, triplet=True, return_latents=True, **dataset_args, **test_args(train_dataset))
    if exclude_objects is not None and data_name == 'causal3d':
        test_dataset = {
            'orig_wo_' + '_'.join([str(o) for o in exclude_objects]): test_dataset
        }
        val_dataset = {
            next(iter(test_dataset.keys())): val_dataset 
        }
        dataset_args.pop('exclude_objects')
        for o in exclude_objects:
            val_dataset[f'exclusive_obj_{o}'] = DataClass(
                                data_folder=data_dir, split='val_indep', single_image=True, triplet=False, return_latents=True, exclude_objects=[i for i in range(7) if i != o], **dataset_args, **test_args(train_dataset))
            test_dataset[f'exclusive_obj_{o}'] = DataClass(
                                data_folder=data_dir, split='test_indep', single_image=True, triplet=False, return_latents=True, exclude_objects=[i for i in range(7) if i != o], **dataset_args, **test_args(train_dataset))
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)
    val_triplet_loader = data.DataLoader(val_triplet_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=False, num_workers=num_workers)
    test_triplet_loader = data.DataLoader(test_triplet_dataset, batch_size=batch_size,
                                  shuffle=False, drop_last=False, num_workers=num_workers)

    print(f'Training dataset size: {len(train_dataset)} / {len(train_loader)}')
    # print(f'Val triplet dataset size: {len(val_triplet_dataset)} / {len(val_triplet_loader)}')
    if isinstance(val_dataset, dict):
        print(f'Val correlation dataset sizes: { {key: len(val_dataset[key]) for key in val_dataset} }')
    else:
        print(f'Val correlation dataset size: {len(val_dataset)}')
    # print(f'Test triplet dataset size: {len(test_triplet_dataset)} / {len(test_triplet_loader)}')
    if isinstance(test_dataset, dict):
        print(f'Test correlation dataset sizes: { {key: len(test_dataset[key]) for key in test_dataset} }')
    else:
        print(f'Test correlation dataset size: {len(test_dataset)}')

    if dataset_name in ('chambers', 'chambers_semi_synth_decoder'):
        datasets = {
            'train': train_dataset,
            'val': val_dataset_indep,
            'test': test_dataset_indep,
            'val_triplet': val_triplet_dataset,
            'test_triplet': test_triplet_dataset
        }
    else:
        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            # 'val_triplet': val_triplet_dataset,
            'val_triplet': train_dataset,
            'test': test_dataset,
            # 'test_triplet': test_triplet_dataset,
            'test_triplet': train_dataset
        }
    if dataset_name in ('chambers', 'chambers_semi_synth_decoder'):
        val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
                                     shuffle=False, drop_last=False,
                                     num_workers=num_workers)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                      shuffle=False, drop_last=False,
                                      num_workers=num_workers)
        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    else:
        data_loaders = {
            'train': train_loader,
            # 'val_triplet': val_triplet_loader,
            'val_triplet': train_loader,
            # 'test_triplet': test_triplet_loader,
            'test_triplet': train_loader
        }
    return datasets, data_loaders, data_name


def print_params(logger_name, model_args):
    num_chars = max(50, 11+len(logger_name))
    print('=' * num_chars)
    print(f'Experiment {logger_name}')
    print('-' * num_chars)
    for key in sorted(list(model_args.keys())):
        print(f'-> {key}: {model_args[key]}')
    print('=' * num_chars)


def train_model(model_class, train_loader, val_loader, 
                test_loader=None,
                logger_name=None,
                max_epochs=200,
                progress_bar_refresh_rate=1,
                check_val_every_n_epoch=1,
                debug=False,
                offline=False,
                op_before_running=None,
                load_pretrained=False,
                root_dir=None,
                files_to_save=None,
                gradient_clip_val=1.0,
                cluster=False,
                callback_kwargs=None,
                seed=42,
                save_last_model=False,
                val_track_metric='val_loss',
                data_dir=None,
                **kwargs):
    trainer_args = {}
    if root_dir is None or root_dir == '':
        root_dir = os.path.join('checkpoints/', model_class.__name__)
    if not (logger_name is None or logger_name == ''):  # TODO: pass wandblogger here
        logger_name = logger_name.split('/')
        logger = pl.loggers.TensorBoardLogger(root_dir,
                                              name=logger_name[0],
                                              version=logger_name[1] if len(logger_name) > 1 else None)
        # Wandb logger
        # logger = pl.loggers.WandbLogger(save_dir=root_dir,
        #                                 name=logger_name[0],  # TODO: this is a different variable in wandb
        #                                 version=logger_name[1] if len(logger_name) > 1 else None)
        trainer_args['logger'] = logger
    if progress_bar_refresh_rate == 0:
        trainer_args['enable_progress_bar'] = False

    if callback_kwargs is None:
        callback_kwargs = dict()
    callbacks = model_class.get_callbacks(exmp_inputs=next(iter(val_loader)), cluster=cluster, 
                                          **callback_kwargs)
    if not debug:
        callbacks.append(
                ModelCheckpoint(save_weights_only=True, 
                                mode="min", 
                                monitor=val_track_metric,
                                save_last=save_last_model,
                                every_n_epochs=check_val_every_n_epoch)
            )
    if debug:
        torch.autograd.set_detect_anomaly(True) 
    trainer = pl.Trainer(default_root_dir=root_dir,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         # gpus=1 if torch.cuda.is_available() else 0,  # TODO: see if we have to change this for mps
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         gradient_clip_val=gradient_clip_val,
                         **trainer_args)
    trainer.logger._default_hp_metric = None

    if files_to_save is not None:
        # log_dir = trainer.logger.log_dir
        log_dir = trainer.logger.save_dir
        os.makedirs(log_dir, exist_ok=True)
        for file in files_to_save:
            if os.path.isfile(file):
                filename = file.split('/')[-1]
                copyfile(file, os.path.join(log_dir, filename))
                print(f'=> Copied {filename}')
            else:
                print(f'=> File not found: {file}')

    # Check whether pretrained model exists. If yes, load it and skip training
    # TODO: might have to change this path to our convention
    pretrained_filename = os.path.join(
        'checkpoints/', model_class.__name__ + ".ckpt")
    # Maybe:
    # pretrained_filename = os.path.join()

    if load_pretrained and os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = model_class.load_from_checkpoint(pretrained_filename)
    else:
        if load_pretrained:
            print("Warning: Could not load any pretrained models despite", load_pretrained)
        pl.seed_everything(seed)  # To be reproducable
        model = model_class(**kwargs)
        if op_before_running is not None:
            op_before_running(model)
        trainer.fit(model, train_loader, val_loader)
        model = model_class.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    if test_loader is not None:
        model_paths = [(trainer.checkpoint_callback.best_model_path, "best")]
        if save_last_model:
            model_paths += [(trainer.checkpoint_callback.last_model_path, "last")]
        for file_path, prefix in model_paths:
            model = model_class.load_from_checkpoint(file_path)
            for c in callbacks:
                if hasattr(c, 'set_test_prefix'):
                    c.set_test_prefix(prefix)
            test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
            test_result = test_result[0]
            print('='*50)
            print(f'Test results ({prefix}):')
            print('-'*50)
            for key in test_result:
                print(key + ':', test_result[key])
            print('='*50)

            # log_file = os.path.join(trainer.logger.log_dir, f'test_results_{prefix}.json')
            log_file = os.path.join(trainer.logger.save_dir,
                                    f'test_results_{prefix}.json')
            with open(log_file, 'w') as f:
                json.dump(test_result, f, indent=4)