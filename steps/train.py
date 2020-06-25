import tqdm
import torch
import torch.nn as nn

from collections import defaultdict
from math import isnan

def run_epoch(models, train_ldr, it, avg_loss, clipping):
    r"""Trains the provided models for an epoch.
    """
    # model_t = 0.0
    # data_t = 0.0
    # end_t = time.time()
    tq = tqdm.tqdm(train_ldr)
    exp_w = 0.99
    losses = defaultdict(list)
    grad_norms = defaultdict(float)
    for batch in tq:
        inputs, labels = list(batch)
        # start_t = time.time()
        for (feature_class, model, optimizer) in models:
            cur_avg_loss = avg_loss[feature_class]
            optimizer.zero_grad()
            loss = model.loss(inputs, labels[feature_class])
            loss.backward(torch.ones_like(loss))
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clipping)
            if(isnan(grad_norm)):
                print("Cannot proceed further--norm of grad is not a number")
                exit(1)

            loss = loss.data[0]
            optimizer.step()
            # prev_end_t = end_t
            # end_t = time.time()
            # model_t += end_t - start_t
            # data_t += start_t - prev_end_t
            cur_avg_loss = exp_w * cur_avg_loss + (1 - exp_w) * loss
            avg_loss[feature_class] = cur_avg_loss.item()
            grad_norms[feature_class] = grad_norm.item()
            losses[feature_class].append(loss.item())
    #        model_time=model_t, data_time=data_t)
        tq.set_postfix(iter=it,
                       avg_losses=dict(avg_loss),
                       grad_norms=dict(grad_norms))
        it += 1
    return it, avg_loss, losses
