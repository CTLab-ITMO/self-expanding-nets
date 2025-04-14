import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error
from senmodel.metrics.nonlinearity_metrics import AbsGradientEdgeMetric
from senmodel.model.utils import *
import torch.optim as optim
from senmodel.metrics.train_metrics import *
from senmodel.metrics.edge_finder import *
import wandb

import matplotlib.pyplot as plt


def train_one_epoch(model, optimizer, criterion, train_loader):
    t0 = time.time()
    model.train()
    train_loss = 0
    for i, (inputs, targets) in enumerate(tqdm(train_loader)):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_time = time.time() - t0
    return train_loss, train_time


def eval_one_epoch(model, criterion, val_loader, task_type):
    '''
    Args:
        model (torch.nn.Module)
        criterion (torch.nn.Module)
        val_loader (torch.utils.data.DataLoader)
        task_type (str): 'regression' or 'classification'.

    Returns:
        tuple: A tuple containing:
            - val_loss (float): Average loss over the validation set.
            - val_accuracy (float) / MSE (float): Accuracy score for classification tasks,
                                   or MSE for regression tasks.
    '''

    model.eval()
    val_loss = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            if task_type == 'classification': preds = torch.argmax(outputs, dim=1)
            else: preds = outputs

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    val_loss /= len(val_loader)

    if task_type == 'classification': metric = accuracy_score(all_targets, all_preds)
    else: metric = mean_squared_error(all_targets, all_preds)

    return val_loss, metric


def edge_replacement_func_new_layer(model, layer, optim, choose_threshold, ef):
    chosen_edges = ef.choose_edges(model, layer, choose_threshold, embed=False, max_limit=True)
    print("Chosen edges:", chosen_edges, len(chosen_edges[0]))
    layer.replace_many(*chosen_edges)
    
    if len(chosen_edges[0]) > 0:
        optim.add_param_group({'params': layer.embed_linears[-1].weight_values})
        optim.add_param_group({'params': layer.weight_values})
    return len(chosen_edges[0])


def edge_deletion_func_new_layer(model, layer, optim, inds, choose_threshold, ef):
    inds_edges_emb = ef.choose_edges(model, layer, choose_threshold, embed=True, max_limit=False)
    inds_edges_exp = ef.choose_edges(model, layer, choose_threshold, embed=False, max_limit=False)
    
    inds_emb, inds_exp = inds
    
    mask_emb = ~torch.any(torch.all(inds_edges_emb.unsqueeze(1) == inds_emb.unsqueeze(0), dim=2), dim=1)
    chosen_edges_emb = inds_edges_emb[mask_emb]

    mask_exp = ~torch.any(torch.all(inds_edges_exp.unsqueeze(1) == inds_exp.unsqueeze(0), dim=2), dim=1)
    chosen_edges_exp = inds_edges_exp[mask_exp]
    
    print("Chosen edges to del emb:", chosen_edges_emb, len(chosen_edges_emb[0]))
    print("Chosen edges to del exp:", chosen_edges_exp, len(chosen_edges_exp[0]))
    
    layer.delete_many(chosen_edges_emb, chosen_edges_exp)

    optim.add_param_group({'params': layer.embed_linears[-1].weight_values})
    optim.add_param_group({'params': layer.weight_values})

    return len(chosen_edges_emb[0]) + len(chosen_edges_exp[0])

def train_sparse_recursive(model, train_loader, val_loader, test_loader, criterion, hyperparams):
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['lr'])
    # tofo device in ef
    ef = EdgeFinder(hyperparams['metric'], val_loader, aggregation_mode='mean', max_to_choose=hyperparams['max_to_choose'])
    non_zero_masks = {}

    replace_epoch = [0]
    val_losses = []
    for epoch in range(hyperparams['num_epochs']):
        train_loss, train_time = train_one_epoch(model, optimizer, criterion, train_loader)
        val_loss, val_accuracy = eval_one_epoch(model, criterion, test_loader, hyperparams['task_type'])
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{hyperparams['num_epochs']}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        if epoch - replace_epoch[-1] > max(hyperparams['delete_after'], hyperparams['min_delta_epoch_replace'], hyperparams['window_size']):
            recent_changes = [abs(val_losses[i] - val_losses[i - 1]) for i in range(-hyperparams['window_size'], 0)]
            avg_change = sum(recent_changes) / hyperparams['window_size']            
            if avg_change < hyperparams['threshold']:
                len_choose = 0
                for layer_name in hyperparams['choose_thresholds'].keys():
                    layer = model.__getattr__(layer_name)
                    
                    # удаление полностью нулевых рёбер
                    if (len(replace_epoch) == 1):
                        metr = ef.calculate_edge_metric_for_dataloader(model, layer, embed=False)
                        mask = metr == 0
                        res = layer.weight_indices[:, mask.nonzero(as_tuple=True)[0]]
                        layer.delete_many(None, res)

                        m = ef.calculate_edge_metric_for_dataloader(model, layer, embed=False)
                        plt.hist(m.cpu().numpy(), bins=100)
                        plt.title(f"ep: {epoch}, initial hist")
                        plt.show()
                    
                    len_choose += edge_replacement_func_new_layer(model, layer, optimizer, hyperparams['choose_thresholds'][layer_name], ef)
                    
                    m = ef.calculate_edge_metric_for_dataloader(model, layer, embed=False)
                    plt.hist(m.cpu().numpy(), bins=100)
                    plt.title(f"ep: {epoch}, after replace")
                    plt.show()
                    
                    non_zero_masks[layer_name] = layer.get_non_zero_params()
                replace_epoch += [epoch]

        # елси хотите удаление, то уберите комментарий
        if epoch - replace_epoch[-1] == hyperparams['delete_after'] and replace_epoch[-1] != 0:
            len_choose = 0
            for layer_name in hyperparams['choose_thresholds'].keys():
                layer = model.__getattr__(layer_name)
                len_choose += edge_deletion_func_new_layer(model, layer, optimizer, non_zero_masks[layer_name], hyperparams['choose_thresholds'][layer_name], ef)
                
                m = ef.calculate_edge_metric_for_dataloader(model, layer, embed=False)
                plt.hist(m.cpu().numpy(), bins=100)
                plt.title(f"ep: {epoch}, after delete")
                plt.show()
        print("aaaaaaa")
        
        params_amount = get_params_amount(model)
        replace_params = 0
        for layer_name in hyperparams['choose_thresholds'].keys():
            layer = model.__getattr__(layer_name)
            
            replace_params += len(ef.choose_edges(model, layer, hyperparams['choose_thresholds'][layer_name])[0])

        logs = {'val loss': val_loss, 'val accuracy': val_accuracy,
                'train loss': train_loss, 'params amount': params_amount,
                'params to replace amount': replace_params, 'train time': train_time,
                'params ratio': (params_amount - replace_params) / params_amount,
                'lr': optimizer.param_groups[0]['lr'], 'acc amount': val_accuracy / params_amount,
                'n_params over train_time': params_amount / train_time,
                'train_time over n_params': train_time / params_amount}

        if (epoch in replace_epoch) and epoch != 0: logs['len_choose'] = len_choose
        else: logs.pop('len_choose', None)
        
        if (epoch + hyperparams['delete_after'] in replace_epoch) and epoch != 0: logs['del_len_choose'] = len_choose
        else: logs.pop('del_len_choose', None)

        # wandb.log(logs)
