import json
import sys
import glob
import os

import torch

from dataloader.dataloader import get_data
from models.trainer import Trainer
from models.trainer_m import TrainerM
from models.generator import Generator
from models.model import Model
from utility.statistics import statistics, statistics_all
from utility.test import test, test_all
from utility.utils import set_reproducibility

def main(fname):
    with open(fname) as fp:
        params = json.load(fp)

    init_seed = params['seed']

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = params["n_gpu"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params["device"] = device

    if not os.path.exists(params["SAVE_FOLDER"]):
        os.mkdir(params["SAVE_FOLDER"])

    if not os.path.exists(os.path.join(params["SAVE_FOLDER"], 'models')):
        os.mkdir(os.path.join(params["SAVE_FOLDER"], 'models'))

    if not os.path.exists(os.path.join(params["SAVE_FOLDER"], 'plots')):
        os.mkdir(os.path.join(params["SAVE_FOLDER"], 'plots'))

    if not os.path.exists(os.path.join(params["SAVE_FOLDER"], 'clustering')):
        os.mkdir(os.path.join(params["SAVE_FOLDER"], 'clustering'))

    if not os.path.exists(os.path.join(params["SAVE_FOLDER"], 'results')):
        os.mkdir(os.path.join(params["SAVE_FOLDER"], 'results'))

    if not os.path.exists(os.path.join(params["SAVE_FOLDER"], 'statistics')):
        os.mkdir(os.path.join(params["SAVE_FOLDER"], 'statistics'))

    if not os.path.exists(os.path.join(params["SAVE_FOLDER"], 'clustering', 'data')):
        os.mkdir(os.path.join(params["SAVE_FOLDER"], 'clustering', 'data'))

    for i in range(params['n_runs']):
        dataset_name = params['dataset_name']
        print(f'Dataset: {dataset_name}')
        params['seed'] = init_seed + 1 # FIX ME
        is_reduction = params['is_reduction']
        is_combined = params['combined']

        if is_reduction:
            params['reduction'] = 1

        seed = params['seed']

        set_reproducibility(seed)

        params['plot_clusters_path'] = os.path.join(params["SAVE_FOLDER"], 'clustering',
                                                    f'clusters_run_{i}_{seed}_{dataset_name}.pdf')
        params['plot_cluster_distribution_path'] = os.path.join(params["SAVE_FOLDER"], 'clustering',
                                                    f'cluster_distribution_run_{i}_{seed}_{dataset_name}.pdf')
        params["path_dendograms"] = os.path.join(params["SAVE_FOLDER"], 'clustering',
                                                    f'dendograms_run_{i}_{seed}_{dataset_name}.pdf')

        perc_split_m = params['perc_split_m']
        split_train, split_val = params["split_train"], params["split_val"]

        embed_dim_m, num_heads_m, num_layers_m = params["embed_dim_m"], params["num_heads_m"], params["num_layers_m"]
        embed_dim_g, num_heads_g, num_layers_g = params["embed_dim_g"], params["num_heads_g"], params["num_layers_g"]

        exp = f'RUN_{i}_SEED_{seed}_DATA_{dataset_name}_M_{embed_dim_m}_H_{num_heads_m}_L_{num_layers_m}_G_{embed_dim_g}_H_{num_heads_g}_L_{num_layers_g}_SPLIT_M_{perc_split_m}_G_T_{split_train}_V_{split_val}_reduction_{is_reduction}_combined_{is_combined}'

        file_path = f'{params["SAVE_FOLDER"]}/results/log_{exp}.txt'

        if params['train_m']:
            open(file_path, 'w').close()
        params['log_path'] = file_path

        best_path_mod, last_path_mod = f"AI_best_model_{exp}.pt", f"AI_last_model_{exp}.pt"
        best_path_gen, last_path_gen = f"Generator_best_model_{exp}.pt", f"Generator_last_model_{exp}.pt"

        params["BEST_PATH_MOD"] = os.path.join(params["SAVE_FOLDER"], 'models', best_path_mod)
        params["LAST_PATH_MOD"] = os.path.join(params["SAVE_FOLDER"], 'models', last_path_mod)
        params["BEST_PATH_GEN"] = os.path.join(params["SAVE_FOLDER"], 'models', best_path_gen)
        params["LAST_PATH_GEN"] = os.path.join(params["SAVE_FOLDER"], 'models', last_path_gen)

        plot_path_loss_g = f'loss_generator_{exp}.pdf'
        plot_path_log_loss_g = f'loss_log_generator_{exp}.pdf'
        plot_path_loss_m = f'loss_AI_{exp}.pdf'
        plot_path_log_loss_m = f'loss_log_AI_{exp}.pdf'
        idx_clustering_path = f'idx_clustering_{exp}.dat'

        params["plot_path_loss_g"] = os.path.join(params["SAVE_FOLDER"], 'plots', plot_path_loss_g)
        params["plot_path_log_loss_g"] = os.path.join(params["SAVE_FOLDER"], 'plots', plot_path_log_loss_g)
        params["plot_path_loss_m"] = os.path.join(params["SAVE_FOLDER"], 'plots', plot_path_loss_m)
        params["plot_path_log_loss_m"] = os.path.join(params["SAVE_FOLDER"], 'plots', plot_path_log_loss_m)
        params['idx_clustering_path'] = os.path.join(params['SAVE_FOLDER'], 'clustering', 'data', idx_clustering_path)

        tmp = f'RUN: {i}, SEED: {seed}, DATA: {dataset_name} --> G params: emb_dim, {embed_dim_g}, n_heads, {num_heads_g}, n_layers, {num_layers_g}; M params: emb_dim, {embed_dim_m}, n_heads, {num_heads_m}, n_layers, {num_layers_m}; M split: {perc_split_m}, train_perc: {split_train}, val_perc: {split_val}, reduction: {is_reduction}, combined: {is_combined}\n'
        print(tmp)

        with open(file_path, 'a') as filehandle:
            filehandle.write(tmp)

        dataloader_g, train_loader_m, val_loader_m, train_loader_g, val_loader_g, params = get_data(params)

        # train Model M
        if params['train_m']:
            model_m = Model(params['n_feats'], params['n_feats'], embed_dim_m, num_heads_m, num_layers_m).to(device)

            trainer_m = TrainerM(model_m, params)
            trainer_m.train(train_loader_m, val_loader_m)
        else:
            model_m = Model(params['n_feats'], params['n_feats'], embed_dim_m, num_heads_m, num_layers_m).to(device)
            model_m.load_state_dict(torch.load(params["BEST_PATH_MOD"]))
            model_m = model_m.to(device)

        if params['train_g']:
            # train Generator G
            generator = Generator(params['n_feats'], params['n_feats'], embed_dim_g, num_heads_g, num_layers_g).to(
                device)

            trainer = Trainer(generator, model_m, params)
            trainer.train(dataloader_g)

        else:
            generator = Generator(params['n_feats'], params['n_feats'], embed_dim_g, num_heads_g, num_layers_g).to(
                device)
            generator.load_state_dict(torch.load(params["BEST_PATH_GEN"]))
            generator = generator.to(device)

        if params['test']:
            # test model
            path_y_prediction_train = os.path.join(params["SAVE_FOLDER"], 'statistics', f'y_preds_train_{exp}.dat')
            path_y_prediction_val = os.path.join(params["SAVE_FOLDER"], 'statistics', f'y_preds_val_{exp}.dat')
            path_y_prediction_g = os.path.join(params["SAVE_FOLDER"], 'statistics', f'y_preds_g_{exp}.dat')

            params['path_y_prediction_train'] = path_y_prediction_train
            params['path_y_prediction_val'] = path_y_prediction_val
            params['path_y_prediction_g'] = path_y_prediction_g

            if params['compute_prediction']:
                test(params, generator, model_m, train_loader_g, val_loader_g)
                test_all(params, generator, model_m, dataloader_g)

            # statistics
            path_out_log_train = os.path.join(params["SAVE_FOLDER"], 'statistics', f'out_log_train_{exp}.pdf')
            path_out_log_val = os.path.join(params["SAVE_FOLDER"], 'statistics', f'out_log_val_{exp}.pdf')
            path_out_log_g = os.path.join(params["SAVE_FOLDER"], 'statistics', f'out_log_g_{exp}.pdf')
            path_plot_scatter = os.path.join(params["SAVE_FOLDER"], 'statistics', f'plot_scatter_{exp}.pdf')

            path_out_log_train_smooth = os.path.join(params["SAVE_FOLDER"], 'statistics', f'out_log_train_smooth_{exp}.pdf')
            path_out_log_val_smooth = os.path.join(params["SAVE_FOLDER"], 'statistics', f'out_log_val_smooth_{exp}.pdf')
            path_out_log_g_smooth = os.path.join(params["SAVE_FOLDER"], 'statistics',
                                                     f'out_log_g_smooth_{exp}.pdf')

            params['path_out_log_train'] = path_out_log_train
            params['path_out_log_val'] = path_out_log_val
            params['path_out_log_g'] = path_out_log_g
            params['path_out_log_train_smooth'] = path_out_log_train_smooth
            params['path_out_log_val_smooth'] = path_out_log_val_smooth
            params['path_out_log_g_smooth'] = path_out_log_g_smooth
            params['path_scatter'] = path_plot_scatter

            if params['compute_stats_all']:
                count_train, count_val = statistics_all(params)
                # with open(file_path, 'a') as filehandle:
                #    filehandle.write(f'ALL G: Copy in train: {count_train}, in val: {count_val}')
            if params['compute_stats_sep']:
                count_train, count_val = statistics(params)
                with open(file_path, 'a') as filehandle:
                    filehandle.write(f'SEP G: Copy in train: {count_train}, in val: {count_val}')

if __name__ == '__main__':
    main(sys.argv[1])
    '''
    all = int(sys.argv[2])
    if all == 2:
        main(sys.argv[1])
    elif all == 0:
        directory_json = "config/"
        files = glob.glob(directory_json + "*.json")
        for file in files:
            main(file)
    elif all == 1:
        directory_json = "config_1/"
        files = glob.glob(directory_json + "*.json")
        for file in files:
            main(file)
    else:
        print('No config files found')
        sys.exit(0)
    '''
