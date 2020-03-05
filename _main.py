import argparse, json, time, random, tqdm, numpy as np, codecs
import torch, torch.nn as nn, torch.optim, torch.utils.data as datautils
import tensorboard_logger as tb

from math import isnan
from models.single import BasicNetwork
from utils.audiodataset import AudioDataset
from utils.preprocessor import Preprocessor
from utils.randombatchsampler import RandomBatchSampler
from utils.utilities import names, save_model, load_model
from utils.stats import compute_cer

def collate_zip(batch):
    return zip(*batch)

def make_loader(lang, data_tsv, preproc, batch_size, num_workers=4):
    dataset = AudioDataset(lang, data_tsv, preproc, batch_size)
    return datautils.DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=RandomBatchSampler(dataset, batch_size),
                            num_workers=num_workers,
                            collate_fn=collate_zip,
                            drop_last=True)

def graves_const(m):
    try:
        if(type(m) == nn.Linear):
            nn.init.uniform_(m.weight,-0.1,0.1)
            nn.init.uniform_(m.bias,-0.1,0.1)
        elif(type(m) == nn.GRU):
            for name, param in m.named_parameters(): 
                if 'weight' in name or 'bias' in name:
                    nn.init.uniform_(param,-0.1,0.1);
        elif(type(m) == nn.Conv2d):
            nn.init.uniform_(m.weight,-0.1,0.1);
            nn.init.uniform_(m.bias,-0.1,0.1);
#            else:
#                print("Didn't work for", type(m))
    except AttributeError:
        print("Didn't work for", type(m))
        pass


clipping = 5

def run_epoch(model, optimizer, train_ldr, it, avg_loss):
    model_t = 0.0
    data_t = 0.0
    end_t = time.time()
    tq = tqdm.tqdm(train_ldr)
    exp_w = 0.99
    for batch in tq:
        start_t = time.time()
        optimizer.zero_grad()
        inputs, labels = list(batch)
        loss = model.loss(inputs, labels)
        loss.backward(torch.ones_like(loss))
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clipping)
        loss = loss.data[0]
        optimizer.step()
        prev_end_t = end_t
        end_t = time.time()
        model_t += end_t - start_t
        data_t += start_t - prev_end_t
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss
        if(isnan(grad_norm)):
            print("Nope!")
            exit(1)
        tb.log_value('train_loss', loss, it)
        tq.set_postfix(iter=it, loss=loss, avg_loss=avg_loss,
            grad_norm=grad_norm)
#        model_time=model_t, data_time=data_t)
        it += 1
    return it, avg_loss

def eval_dev(model, ldr, preproc):
    losses = []
    all_preds = []
    all_labels = []
    model.set_eval()
    for batch in tqdm.tqdm(ldr):
        inputs, labels = list(batch)
        with torch.no_grad():
            preds = model.infer(inputs, labels)
            loss = model.loss(inputs, labels)
        losses.append(loss.data[0])
        all_preds.extend(preds)
        all_labels.extend(labels)
    model.set_train()
    loss = sum(losses)/len(losses)
    results = [(preproc.decode(l), preproc.decode(p)) for l, p in zip(all_labels, all_preds)]
    with codecs.open('results_dev.tsv','w','utf-8') as f:
        f.write(str(results))
    cer = compute_cer(results)
    print("Dev loss {:.3f}, cer {:.3f}".format(loss, cer))
    return loss, cer

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
    tb.configure(config["save_path"])

    enable_cuda = torch.cuda.is_available()
    if enable_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False
    
    opt_cfg = config["optimizer"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    batch_size = opt_cfg["batch_size"]

    preproc = Preprocessor(data_cfg["lang"], data_cfg["train_set"])
    train_ldr = make_loader(data_cfg["lang"], data_cfg["train_set"], preproc, batch_size)
    dev_ldr = make_loader(data_cfg["lang"], data_cfg["dev_set"], preproc, batch_size)
    model_class = BasicNetwork # select_model(model_cfg["class"])
    model = model_class(preproc.input_dim, preproc.phone_size, model_cfg)

    model.apply(graves_const)
    model.cuda() if enable_cuda else model.cpu()
    assert model.is_cuda

    print("lr =",opt_cfg["learning_rate"],
            "and momentum =",opt_cfg["momentum"],
            "and clip threshold =",clipping)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt_cfg["learning_rate"],
                                momentum=opt_cfg["momentum"])
    run_state = (0, 0)
    best_so_far = float("inf")
    for e in range(opt_cfg["epochs"]):
        start = time.time()
        run_state = run_epoch(model, optimizer, train_ldr, *run_state)
        print("{} took {:.2f}s".format(e,time.time()-start))
        dev_loss, dev_cer = eval_dev(model,dev_ldr,preproc)
        tb.log_value("dev_loss", dev_loss, e)
        tb.log_value("dev_cer", dev_cer, e)
        save_model(model, preproc, config["save_path"])
        if(dev_cer < best_so_far):
            best_so_far = dev_cer
            save_model(model, preproc, config["save_path"], tag="best")
            if(e != 0):
                print('Halving learning rate:')
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 2
        elif((e+1) % 4 == 0):
            print('Halving learning rate:')
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / 2
