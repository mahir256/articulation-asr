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
from torch.utils.tensorboard import SummaryWriter

import models
from steps.train import run_epoch
from steps.eval_dev import eval_dev
from steps.loader import make_loader
from utils.nn_manipulation import graves_const, learning_rate_adjustment
from utils.preprocessor import Preprocessor
from utils.model_io import names, save_model, load_model

def feature_class_outputs(feature_classes):
    return [(feature_class, preproc.phone_size(feature_class)) for feature_class in feature_classes]

def construct_basic_network(feature_class, preproc, model_cfg, data_cfg, opt_cfg, enable_cuda=True):
    feature_classes = set([feature_class])
    if(data_cfg.get("joint_fc_out",False)):
        feature_classes |= set(data_cfg["joint_fc_out"])
    feature_classes = list(feature_classes)
    model = models.BasicNetwork(preproc.input_dim, feature_class_outputs(feature_classes), model_cfg)

    model.apply(graves_const)
    model.cuda() if enable_cuda else model.cpu()
    assert model.is_cuda

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=schedule_cfg["learning_rate"],
                                momentum=schedule_cfg["momentum"])
    return feature_class, model, optimizer

def construct_basic_network_set(preproc, data_cfg, model_cfg, opt_cfg):
    model_list = []
    for feature_class in data_cfg["feature_classes"]:
        model_list.append(construct_basic_network(feature_class, preproc, model_cfg, data_cfg, opt_cfg))
    return model_list

def construct_prog_network_set(preproc, data_cfg, model_cfg, opt_cfg, enable_cuda=True):
    priornets = []
    for feature_class in data_cfg["feature_classes"]:
        model, _ = load_model(data_cfg["basis_path"],feature_class,"best")
        priornets.append(model)
    _, currentnet, optimizer = construct_basic_network("phones", preproc, model_cfg, data_cfg, opt_cfg)
    prognet = models.ProgressiveNetwork(preproc.input_dim, preproc.phone_size(), model_cfg, priornets, currentnet)
    prognet.cuda() if enable_cuda else prognet.cpu()
    return [("phones", prognet, optimizer)]

def select_preprocessor_features(model_cfg, data_cfg):
    featurelist = ["phones"] if (model_cfg["class"] == "ProgressiveNetwork") else data_cfg["feature_classes"]
    if(data_cfg.get("joint_fc_out",False)):
        featurelist.extend(data_cfg["joint_fc_out"])
    return featurelist

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)

def construct_model(preproc, data_cfg, model_cfg, opt_cfg):
    if(model_class == "BasicNetwork"):
        model_list = construct_basic_network_set(preproc, data_cfg, model_cfg, opt_cfg)
    elif(model_class == "ProgressiveNetwork"):
        model_list = construct_prog_network_set(preproc, data_cfg, model_cfg, opt_cfg)
    return model_list

def get_preprocessors(lang_in, data_cfg, model_cfg):
    preproc = Preprocessor(lang_in, data_cfg["train_set"], select_preprocessor_features(model_cfg, data_cfg))
    train_ldr = make_loader(lang_in, data_cfg["train_set"], preproc, batch_size)
    dev_ldr = make_loader(lang_in, data_cfg["dev_set"], preproc, batch_size)
    return preproc, train_ldr, dev_ldr

def raw_filename(feature_class,tag,iteration):
    return "_".join([feature_class,tag,str(iteration)])

def export_raw(entries,filename):
    output_path = os.path.join(config["save_path"],filename)
    with codecs.open(output_path,'w','utf-8') as f:
        for entry in entries:
            f.write(str(entry)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CTC-based model training.')
    parser.add_argument("config",help="Training configuration file.")
    parser.add_argument("--deterministic",action="store_true",
                        help="Deterministic mode (i.e. no cudnn).")
    parser.add_argument("--tensorboard",action="store_true",
                        help="Generates logs suitable for use with Tensorboard.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    set_seeds(config["seed"])

    writer = SummaryWriter() if args.tensorboard else None

    opt_cfg = config["optimizer"]
    data_cfg = config["data"]
    model_cfg = config["model"]

    lang_in = data_cfg["lang"]

    batch_size = opt_cfg["batch_size"]
    schedule_cfg = opt_cfg["schedule"]
    epoch_count = opt_cfg["epochs"]

    enable_cuda = torch.cuda.is_available()
    if enable_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False
    
    print("learning rate =",schedule_cfg["learning_rate"],
            "; momentum =",schedule_cfg["momentum"],
            "; clip threshold =",schedule_cfg["clipping"])
    
    model_class = model_cfg["class"]

    preproc, train_ldr, dev_ldr = get_preprocessors(lang_in, data_cfg, model_cfg)

    model_list = construct_model(preproc, data_cfg, model_cfg, opt_cfg)

    itnum = 0
    avg_loss = defaultdict(float)
    best_so_far = defaultdict(lambda: float("inf"))
    dev_losses = defaultdict(list)
    dev_cers = defaultdict(list)

    for e in range(epoch_count):
        start = time.time()

        itnum, avg_loss, losses = run_epoch(model_list, train_ldr, itnum, avg_loss, schedule_cfg["clipping"])

        print("{} took {:.2f}s".format(e,time.time()-start))

        dev_loss, dev_cer, all_results = eval_dev(model_list,dev_ldr,preproc)

        for (feature_class, model, optimizer) in model_list:
            if(writer):
                writer.add_scalar('/'.join([feature_class,"loss"]), dev_loss, e)
                writer.add_scalar('/'.join([feature_class,"cer"]), dev_cer, e)

            # TODO: adjust to use torch.utils.tensorboard somehow
            export_raw(losses[feature_class],raw_filename(feature_class,'losses',e))
            export_raw(all_results[feature_class],raw_filename(feature_class,'results',e))

            save_model(model, preproc, config["save_path"], feature_class, tag="last")
            if(dev_cer[feature_class] < best_so_far[feature_class]):
                save_model(model, preproc, config["save_path"], feature_class, tag="best")
                best_so_far[feature_class] = dev_cer[feature_class]
            if(dev_cer[feature_class] > best_so_far[feature_class]):
                print("Adjusting learning rate for",feature_class)
                learning_rate_adjustment(optimizer)
