import tqdm
import torch
from collections import defaultdict

from utils.stats import compute_cer

def eval_dev(models, ldr, preproc):
    r"""Evaluates the models on the dev set and returns the CERs
    for all models.
    """

    losses = defaultdict(list)
    loss_out = {}
    cer = {}

    all_preds = defaultdict(list)
    all_labels = defaultdict(list)
    all_results = {}

    for (_, model, _) in models:
        model.set_eval()

    for batch in tqdm.tqdm(ldr):
        inputs, labels = list(batch)

        for (feature_class, model, _) in models:
            with torch.no_grad():
                preds = model.infer(inputs, labels[feature_class])
                loss = model.loss(inputs, labels[feature_class])
            losses[feature_class].append(loss[feature_class].data[0].item())
            all_preds[feature_class].extend(preds[feature_class])
            all_labels[feature_class].extend(labels[feature_class])
            # TODO: make more general
            if("phones" in loss and feature_class != "phones"):
                all_preds["phones"].extend(preds["phones"])
                all_labels["phones"].extend(labels["phones"])

    for (feature_class, model, _) in models:
        model.set_train()
        loss_out[feature_class] = sum(losses[feature_class])/len(losses[feature_class])
        results = []
        for l, p in zip(all_labels[feature_class], all_preds[feature_class]):
            results.append((preproc.decode(l,feature_class), preproc.decode(p,feature_class)))
        all_results[feature_class] = results
        cer[feature_class] = compute_cer(results)
        print("For {}: dev loss {:.3f}, cer {:.3f}".format(feature_class, loss_out[feature_class], cer[feature_class]))
        # TODO: make more general
        if("phones" in all_preds and feature_class != "phones"):
            loss_out["phones"] = sum(losses["phones"])/len(losses["phones"])
            results = []
            for l, p in zip(all_labels["phones"], all_preds["phones"]):
                results.append((preproc.decode(l,"phones"), preproc.decode(p,"phones")))
            all_results["phones"] = results
            cer["phones"] = compute_cer(results)
            print("For {}: dev loss {:.3f}, cer {:.3f}".format("phones", loss_out["phones"], cer["phones"]))

    return loss_out, cer, all_results

