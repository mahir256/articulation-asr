import argparse
import json
import time
import random
import codecs
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim

import models
from steps.train import run_epoch
from steps.eval_dev import eval_dev
from steps.loader import make_loader
from utils.nn_manipulation import graves_const, learning_rate_adjustment
from utils.preprocessor import Preprocessor
from utils.model_io import names, save_model, load_model

def construct_basic_network(feature_class, preproc, model_cfg, opt_cfg, enable_cuda=True):
    model = models.BasicNetwork(preproc.input_dim, preproc.phone_size(feature_class), model_cfg)

    model.apply(graves_const)
    model.cuda() if enable_cuda else model.cpu()
    assert model.is_cuda

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt_cfg["learning_rate"],
                                momentum=opt_cfg["momentum"])
    return feature_class, model, optimizer

def construct_basic_network_set(preproc, data_cfg, model_cfg, opt_cfg):
    model_list = []
    for feature_class in data_cfg["feature_classes"]:
        model_list.append(construct_basic_network(feature_class, preproc, model_cfg, opt_cfg))
    return model_list

def construct_prog_network_set(preproc, data_cfg, model_cfg, opt_cfg, enable_cuda=True):
    priornets = []
    for feature_class in data_cfg["feature_classes"]:
        model, _ = load_model(data_cfg["basis_path"],"_".join([feature_class,"best"]))
        priornets.append(model)
    _, currentnet, optimizer = construct_basic_network("phones", preproc, model_cfg, opt_cfg)
    prognet = models.ProgressiveNetwork(preproc.input_dim, preproc.phone_size(), model_cfg, priornets, currentnet)
    prognet.cuda() if enable_cuda else prognet.cpu()
    return [("phones", prognet, optimizer)]

def select_preprocessor_features(model_cfg, data_cfg):
    return ["phones"] if (model_cfg["class"] == "ProgressiveNetwork") else data_cfg["feature_classes"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CTC-based model training.')
    parser.add_argument("config",help="Training configuration file.")
    parser.add_argument("--deterministic",default=False,action="store_true",
                        help="Deterministic mode (i.e. no cudnn).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    enable_cuda = torch.cuda.is_available()
    if enable_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False
    
    opt_cfg = config["optimizer"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    batch_size = opt_cfg["batch_size"]

    preproc = Preprocessor(data_cfg["lang"], data_cfg["train_set"], select_preprocessor_features(model_cfg, data_cfg))
    train_ldr = make_loader(data_cfg["lang"], data_cfg["train_set"], preproc, batch_size)
    dev_ldr = make_loader(data_cfg["lang"], data_cfg["dev_set"], preproc, batch_size)

    print("learning rate =",opt_cfg["learning_rate"],
            "; momentum =",opt_cfg["momentum"],
            "; clip threshold =",opt_cfg["clipping"])
    
    model_class = model_cfg["class"]

    if(model_class == "BasicNetwork"):
        model_list = construct_basic_network_set(preproc, data_cfg, model_cfg, opt_cfg)
    elif(model_class == "ProgressiveNetwork"):
        model_list = construct_prog_network_set(preproc, data_cfg, model_cfg, opt_cfg)

    it = 0
    avg_loss = defaultdict(float)
    best_so_far = defaultdict(lambda: float("inf"))
    dev_losses = defaultdict(list)
    dev_cers = defaultdict(list)
    for e in range(opt_cfg["epochs"]):
        start = time.time()
        it, avg_loss, losses = run_epoch(model_list, train_ldr, it, avg_loss, opt_cfg["clipping"])
        print("{} took {:.2f}s".format(e,time.time()-start))
        dev_loss, dev_cer, all_results = eval_dev(model_list,dev_ldr,preproc)
        print("{} {}".format(dev_loss, dev_cer))
        # tb.log_value("dev_loss", dev_loss, e)
        # tb.log_value("dev_cer", dev_cer, e)
        for (feature_class, model, optimizer) in model_list:
            loss_filename = os.path.join(config["save_path"],"_".join([feature_class,str(e),'losses']))
            with codecs.open(loss_filename,'w','utf-8') as f:
                for _loss in losses[feature_class]:
                    f.write(str(_loss)+'\n')
            scripts_filename = os.path.join(config["save_path"],"_".join([feature_class,str(e),'scripts']))
            with codecs.open(scripts_filename,'w','utf-8') as f:
                for (label, pred) in all_results[feature_class]:
                    f.write(str(pred)+'\t'+str(label)+'\n')
            save_model(model, preproc, config["save_path"], tag=feature_class)
            if(dev_cer[feature_class] < best_so_far[feature_class]):
                save_model(model, preproc, config["save_path"], tag=feature_class+"_best")
            print("Adjusting learning rate for",feature_class)
            learning_rate_adjustment(optimizer)
