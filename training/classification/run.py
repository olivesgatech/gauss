import argparse
import toml
import shutil
import os
import random
import numpy as np
import torch
from config import BaseConfig
from training.classification.trainer import ClassificationTrainer
from stattracking.utils.classificationtracker import get_uspec_inputs


def main(cfg: BaseConfig):
    trainer = ClassificationTrainer(cfg)
    stat_inputs = get_uspec_inputs(cfg)

    for seed in range(cfg.run_configs.start_seed, cfg.run_configs.end_seed):
        epochs = cfg.classification.epochs
        for i in range(epochs):
            trainer.training(i, save_checkpoint=True, track_summaries=True)
            if cfg.classification.track_statistics:
                for elem in stat_inputs:
                    print(elem[1])
                    _, _ = trainer.testing(0, alternative_loader_struct=elem[2])

        # gather train events
        if cfg.classification.track_statistics:
            path = cfg.run_configs.ld_folder_name + '/events/train/'
            if not os.path.exists(path):
                os.makedirs(path)
            trainer.train_statistics.save_statistics(path, f'/seed{str(seed)}/')

        cur_row = []
        names = []
        for elem in stat_inputs:
            print(elem[1])
            preds, cur_acc = trainer.testing(0, alternative_loader_struct=elem[2],
                                             track_embeddings=cfg.classification.track_embeddings,
                                             name=elem[1])
            # predictions
            if cfg.classification.track_predictions:
                cur_row.append(cur_acc)
                names.append(elem[1])
                path = cfg.run_configs.ld_folder_name + '/predictions/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/predictions_seed' + str(seed) + '.npy', np.squeeze(preds['predictions']))

            # embeddings
            if cfg.classification.track_embeddings:
                cur_row.append(cur_acc)
                names.append(elem[1])

                # grad embeddings
                path = cfg.run_configs.ld_folder_name + '/gradembeddings/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/embeddings_seed' + str(seed) + '.npy', np.squeeze(preds['grad_embeddings']))

                # non grad embeddings (for e.g. coreset)
                path = cfg.run_configs.ld_folder_name + '/nongradembeddings/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/embeddings_seed' + str(seed) + '.npy', np.squeeze(preds['nongrad_embeddings']))

            # probabilities
            if cfg.classification.track_probs:
                cur_row.append(cur_acc)
                names.append(elem[1])
                path = cfg.run_configs.ld_folder_name + '/probabilities/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/probabilities_seed' + str(seed) + '.npy', np.squeeze(preds['probs']))

            # forgetting events and switch events
            if cfg.classification.track_statistics and elem[1] != 'train':
                path = cfg.run_configs.ld_folder_name + '/events/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                tracker = elem[2][1]
                tracker.save_statistics(path, f'/seed{str(seed)}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/USPEC-LD/example_config.toml')

    args = parser.parse_args()
    # set random seeds deterministicly to 0
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    main(configs)
    shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + 'parameters.toml')
