'''
Fine-tunes ESM Transformer models with labelled data.
'''

import argparse
import functools
from itertools import chain
import os
import pathlib
import random

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel
from esm import BatchConverter, pretrained

from utils import read_fasta
from utils.metric_utils import spearman, topk_mean, r2, hit_rate, aucroc, ndcg
from utils.esm_utils import RandomCropBatchConverter, CSVBatchedDataset

mse_criterion = torch.nn.MSELoss(reduction='mean')
ce_criterion = torch.nn.CrossEntropyLoss(reduction='none')


def create_parser():
    parser = argparse.ArgumentParser(
        description="Supervised finetuning for sequences in a FASTA file"  # noqa
    )
    parser.add_argument(
        "csv_file",
        type=pathlib.Path,
        help="csv file for labeled data",
    )
    parser.add_argument(
        "wt_fasta_file",
        type=pathlib.Path,
        help="fasta file for WT sequence",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory",
    )
    parser.add_argument(
        "--model_location",
        type=str,
        help="initial model location",
        default='/mnt/esm_weights/esm1b/esm1b_t33_650M_UR50S.pt',
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed"
    )
    parser.add_argument(
        "--n_train", type=int, default=-1, help="training data size"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="validation split"
    )
    parser.add_argument(
        "--n_test", type=int, default=-1, help="test data size"
    )
    parser.add_argument(
        "--toks_per_batch", type=int, default=512, help="maximum batch size"
    )
    parser.add_argument(
        "--max_len", type=int, default=500, help="maximum seq len"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5, help="lr"
    )
    parser.add_argument(
        "--save_model", dest="save_model", action="store_true"
    )
    parser.add_argument(
        "--train_on_single", dest="train_on_single", action="store_true"
    )
    parser.add_argument(
        "--train_on_all", dest="train_on_single", action="store_false"
    )
    parser.set_defaults(save_model=False, train_on_single=True)

    return parser


def step(model, labels, toks, wt_toks, mask_idx):
    labels = torch.tensor(labels)    
    if torch.cuda.is_available():
        labels = labels.to(device="cuda", non_blocking=True)
    predictions = predict(model, toks, wt_toks, mask_idx)
    loss = mse_criterion(predictions, labels)
    return loss, predictions


def predict(model, toks, wt_toks, mask_idx):    
    if torch.cuda.is_available():
        toks = toks.to(device="cuda", non_blocking=True)
    wt_toks_rep = wt_toks.repeat(toks.shape[0], 1)
    mask = (toks != wt_toks)
    masked_toks = torch.where(mask, mask_idx, toks)
    out = model(masked_toks, return_contacts=False)
    logits = out["logits"]
    logits_tr = logits.transpose(1, 2)  # [B, E, T]
    ce_loss_mut = ce_criterion(logits_tr, toks)   # [B, E]
    ce_loss_wt = ce_criterion(logits_tr, wt_toks_rep)
    ll_diff_sum = torch.sum(
        (ce_loss_wt - ce_loss_mut) * mask, dim=1, keepdim=True)  # [B, 1]
    return ll_diff_sum[:, 0]


def main(args):
    model_data = torch.load(args.model_location, map_location='cpu')
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    #repr_layers = [model.num_layers]   # extract last layer
    #toplinear = torch.nn.Linear(model.args.embed_dim, 1)
    #toplinear = torch.nn.Linear(1, 1)

    batch_converter = BatchConverter(alphabet)

    mask_idx = torch.tensor(alphabet.mask_idx)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wt_seq = read_fasta(args.wt_fasta_file)[0]
    _, _, wt_toks = batch_converter([('WT', wt_seq)])

    if torch.cuda.is_available():
        model = model.cuda()
        #toplinear = toplinear.cuda()
        mask_idx = mask_idx.cuda()
        wt_toks = wt_toks.cuda()
        print("Transferred model to GPU")

    df_full = shuffle(pd.read_csv(args.csv_file), random_state=args.seed)
    print(f"Read {args.csv_file} with {len(df_full)} sequences")

    if args.n_test == -1:
        args.n_test = int(len(df_full) * 0.2)
    df_test = df_full[-args.n_test:]
    df_trainval = df_full.drop(df_test.index)

    if args.train_on_single and 'n_mut' in df_full.columns:
         df_trainval = df_trainval[df_trainval.n_mut <= 1]  
    if args.n_train == -1:
        args.n_train = int(len(df_trainval))
    if args.n_train > len(df_trainval):
        print(f"Insufficient data")
        return
    n_val = int(args.n_train * args.val_split)
    df_train = df_trainval[:args.n_train-n_val]
    df_val = df_trainval[args.n_train-n_val:args.n_train]

    train_dataset = CSVBatchedDataset.from_dataframe(df_train)
    val_dataset = CSVBatchedDataset.from_dataframe(df_val)
    test_dataset = CSVBatchedDataset.from_dataframe(df_test)

    train_batches = train_dataset.get_batch_indices(
        args.toks_per_batch, extra_toks_per_seq=1)
    val_batches = val_dataset.get_batch_indices(
        args.toks_per_batch, extra_toks_per_seq=1)
    test_batches = test_dataset.get_batch_indices(
        args.toks_per_batch, extra_toks_per_seq=1)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
            collate_fn=batch_converter, batch_sampler=train_batches)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
            collate_fn=batch_converter, batch_sampler=val_batches)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
            collate_fn=batch_converter, batch_sampler=test_batches)

    #optimizer = torch.optim.Adam(
    #    chain(model.parameters(), toplinear.parameters()), lr=args.learning_rate)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate)
    train_loss = np.zeros(args.epochs+1)
    val_loss = np.zeros(args.epochs+1)
    val_spearman = np.zeros(args.epochs+1)
    best_val_spearman = None

    for epoch in range(args.epochs+1):
        # Train
        if epoch > 0:
            for batch_idx, (labels, strs, toks) in enumerate(train_data_loader):
                if batch_idx % 100 == 0:
                    print(
                        f"Processing {batch_idx + 1} of {len(train_batches)} "
                        f"batches ({toks.size(0)} sequences)"
                    )
                optimizer.zero_grad()
                loss, _ = step(model, labels, toks, wt_toks, mask_idx)
                loss.backward()
                optimizer.step()
                train_loss[epoch] += loss.to('cpu').item()
            train_loss[epoch] /= float(len(train_data_loader))

        # Validation 
        model_eval = model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(val_data_loader):
                loss, predictions = step(
                    model, labels, toks, wt_toks, mask_idx)
                y_pred.append(predictions.to('cpu').numpy())
                y_true.append(labels)
                val_loss[epoch] += loss.to('cpu').item()
        val_loss[epoch] /= float(len(val_data_loader))
        print('epoch %d, train loss: %.3f, val loss: %.3f' % (
            epoch + 1, train_loss[epoch], val_loss[epoch]))
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        val_spearman[epoch] = spearman(y_pred, y_true)
        print(f'Val Spearman correlation {val_spearman[epoch]}')

        if best_val_spearman is None or val_spearman[epoch] > best_val_spearman:
            best_val_spearman = val_spearman[epoch]
            model_data["model"] = model.state_dict() 
            #model_data["toplinear"] = toplinear.state_dict()
            torch.save(model_data, os.path.join(args.output_dir, 'model_data.pt'))

    np.savetxt(os.path.join(args.output_dir, 'loss_trajectory_train.npy'), train_loss)
    np.savetxt(os.path.join(args.output_dir, 'loss_trajectory_val.npy'), val_loss)
    np.savetxt(os.path.join(args.output_dir, 'spearman_trajectory_val.npy'), val_spearman)

    # Load best saved model
    model, alphabet = pretrained.load_model_and_alphabet(
        os.path.join(args.output_dir, 'model_data.pt'))
    if torch.cuda.is_available():
        model = model.cuda()
    model_eval = model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(test_data_loader):
            predictions = predict(model, toks, wt_toks, mask_idx)
            y_pred.append(predictions.to('cpu').numpy())
            y_true.append(labels)
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    print(f'Test Spearman correlation {spearman(y_pred, y_true)}')
    df_test = df_test.copy()
    df_test['pred'] = y_pred
    metric_fns = {
        'spearman': spearman,
        'ndcg': ndcg,
    }
    results_dict = {k: mf(df_test.pred.values, df_test.log_fitness.values)
            for k, mf in metric_fns.items()}
    results_dict.update({
        'predictor': 'esm_finetune',
        'n_train': args.n_train,
        'seed': args.seed,
        'epochs': args.epochs,
    })
    if 'n_mut' in df_test.columns:
        max_n_mut = min(df_test.n_mut.max(), 5)
        for j in range(1, max_n_mut+1):
            y_pred = df_test[df_test.n_mut == j].pred.values
            y_true = df_test[df_test.n_mut == j].log_fitness.values
            results_dict.update({
                    f'{k}_{j}mut': mf(y_pred, y_true)
                    for k, mf in metric_fns.items()})
    results = pd.DataFrame(columns=sorted(results_dict.keys()))
    results = results.append(results_dict, ignore_index=True)
    results.to_csv(os.path.join(args.output_dir, 'metrics.csv'),
        mode='w', index=False, columns=sorted(results.columns.values))

    if not args.save_model:
        os.remove(os.path.join(args.output_dir, 'model_data.pt'))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
