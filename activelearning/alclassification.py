import random
import toml
import os
import shutil
import argparse
import torch
import numpy as np
import pandas as pd
from config import BaseConfig
from data.datasets.classification.common.aquisition import get_dataset
from activelearning.classification.trainer import ActiveLearningClassificationTrainer
from activelearning.qustrategies import get_sampler
from stattracking.utils.classificationtracker import get_uspec_inputs


def main(cfg: BaseConfig):
    np.random.seed(cfg.active_learning.init_seed)

    # get all relevant statistics for the dataset
    train_configs = get_dataset(cfg)
    n_pool = train_configs.data_config.train_len
    nquery = cfg.active_learning.n_query
    nstart = cfg.active_learning.n_start
    nend = cfg.active_learning.n_end
    start_idxs = np.arange(n_pool)[np.random.permutation(n_pool)][:nstart]

    print('Saving in: ' + cfg.run_configs.ld_folder_name)

    if nend < n_pool:
        nrounds = int((nend - nstart) / nquery)
        print('Rounds: %d' % nrounds)
    else:
        nrounds = int((n_pool - nstart) / nquery) + 1
        print('Number of end samples too large! Using total number of samples instead. Rounds: %d Total Samples: %d' %
              (nrounds, n_pool))

    for i in range(cfg.run_configs.start_seed, cfg.run_configs.end_seed):
        # set random seeds
        random.seed(i)
        torch.manual_seed(i)

        sampler = get_sampler(cfg=cfg, n_pool=n_pool, start_idxs=start_idxs)
        accuracies = []
        uspec_inputs = get_uspec_inputs(cfg)

        for round in range(nrounds):
            trainer = ActiveLearningClassificationTrainer(cfg)
            trainer.update_loader(sampler.idx_current.astype(int), np.squeeze(np.argwhere(sampler.total_pool == 0)))

            acc = 0.0
            epoch = 1

            while acc < cfg.active_learning.convergence_acc:
                # reset model if not converging
                if epoch % 80 == 0 and acc < 20.0:
                    print('Model not converging! Resetting now!')
                    del(trainer)
                    trainer = ActiveLearningClassificationTrainer(cfg)
                    trainer.update_loader(sampler.idx_current.astype(int),
                                          np.squeeze(np.argwhere(sampler.total_pool == 0)))
                    epoch = 1
                acc = trainer.training(epoch)
                print('Round: %d' % round)
                print('Seed: %d' % i)

                # perform sampling action
                sampler.action(trainer)
                epoch += 1

            try:
                new_idxs = sampler.query(nquery, trainer)
                sampler.update(new_idxs)

                # save query idxs
                if cfg.active_learning.save_query_idxs:
                    path = cfg.run_configs.ld_folder_name + '/round' + str(round) + '/queryidxs/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    np.save(path + '/newidxs_seed' + str(i) + '.npy', np.squeeze(new_idxs))

                # save training forgetting statistics
                if cfg.classification.track_statistics:
                    folder = cfg.run_configs.ld_folder_name + f'/round{round}/seed{i}/'
                    trainer.save_statistics(folder)
            except:
                if round != nrounds - 1:
                    raise Exception('Sampler updating failing for other rounds than last!')
                print(f'Skipping final sampler update in last round!')

            cur_row = []
            names = []
            for elem in uspec_inputs:
                print(elem[1])
                preds, cur_acc = trainer.testing(0, alternative_loader_struct=elem[2], name=elem[1])
                cur_row.append(cur_acc)
                names.append(elem[1])
                path = cfg.run_configs.ld_folder_name + '/round' + str(round) + '/uspec_statistics/' + elem[1]
                if not os.path.exists(path):
                    os.makedirs(path)
                np.save(path + '/predictions_seed' + str(i) + '.npy', np.squeeze(preds['predictions']))
            accuracies.append(cur_row)

            del trainer

        acc_df = pd.DataFrame(np.array(accuracies), columns=names)
        print(acc_df)
        acc_df.to_excel(cfg.run_configs.ld_folder_name + '/al_seed' + str(i) + '.xlsx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run classification workfow for LD tracking')
    parser.add_argument('--config', help='Path to input config file', type=str,
                        default='~/PycharmProjects/alnfr/example_config.toml')

    args = parser.parse_args()
    configs = toml.load(os.path.expanduser(args.config))
    configs = BaseConfig(configs)
    main(configs)
    shutil.copyfile(os.path.expanduser(args.config), configs.run_configs.ld_folder_name + 'parameters.toml')
