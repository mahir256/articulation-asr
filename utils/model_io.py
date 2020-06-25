import os, torch, dill

def names(path, tag):
    r"""
    """
    tag = tag + "_" if tag else ""
    model = os.path.join(path, tag + "model")
    preprocessor = os.path.join(path, tag + "preprocessor.pyc")
    return model, preprocessor

def save_model(model, preprocessor, path, tag=""):
    r"""
    """
    model_name, preprocessor_name = names(path, tag)
    torch.save(model, model_name)
    with open(preprocessor_name, 'wb') as f:
        dill.dump(preprocessor, f)

def load_model(path, tag=""):
    r"""
    """
    model_name, preprocessor_name = names(path, tag)
    model = torch.load(model_name)
    with open(preprocessor_name, 'rb') as f:
        preprocessor = dill.load(f)
    return model, preprocessor