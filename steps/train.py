import tqdm
import torch
import torch.nn as nn

from collections import defaultdict
from math import isnan

def run_epoch(models, train_ldr, it, avg_loss, clipping):
    r"""Trains the provided models for an epoch.
    """
    # TODO: make model timing wholly optional
    # model_t = 0.0
    # data_t = 0.0
    # end_t = time.time()
    tq = tqdm.tqdm(train_ldr)
    exp_w = 0.99
    overall_losses = defaultdict(list)
    grad_norms = defaultdict(float)
    for batch in tq:
        inputs, labels = list(batch)
        # start_t = time.time()
        for (feature_class, model, optimizer) in models:
            cur_avg_loss = avg_loss[feature_class]
            optimizer.zero_grad()
            losses = model.loss(inputs, labels[feature_class])
            combined_losses = torch.zeros_like(losses[feature_class])
            for feature_class, loss in losses.items():
                combined_losses += loss
            combined_losses.backward(torch.ones_like(combined_losses))
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clipping)
            if(isnan(grad_norm)):
                raise ValueError("Norm of grad is not a number")

            combined_losses = combined_losses.data[0]
            optimizer.step()
            # prev_end_t = end_t
            # end_t = time.time()
            # model_t += end_t - start_t
            # data_t += start_t - prev_end_t
            cur_avg_loss = exp_w * cur_avg_loss + (1 - exp_w) * combined_losses
            avg_loss[feature_class] = cur_avg_loss.item()
            grad_norms[feature_class] = grad_norm.item()
            overall_losses[feature_class].append(combined_losses.item())
        tq.set_postfix(avg_losses={feature: round(value,5) for feature, value in avg_loss.items()},
                       grad_norms={feature: round(value,5) for feature, value in grad_norms.items()})
    #        model_time=model_t, data_time=data_t)
        it += 1
    return it, avg_loss, overall_losses
