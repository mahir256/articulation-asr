import os, torch, dill

def names(path, feature_class=None, tag=None):
    r"""Generates consistent file paths for the export of models.
    """
    feature_class = str(feature_class) if (feature_class is not None) else ""
    tag = str(tag) if (tag is not None) else ""

    name_prefix = "_".join([feature_class,tag])
    model_name = "_".join([name_prefix,"model"])
    preprocessor_name = "_".join([name_prefix,"preprocessor.pyc"])

    model = os.path.join(path, model_name)
    preprocessor = os.path.join(path, preprocessor_name)
    return model, preprocessor

def save_model(model, preprocessor, path, feature_class, tag="best"):
    r"""Saves a model and a preprocessor (the former with PyTorch, the latter with dill).
    """
    model_name, preprocessor_name = names(path, feature_class, tag)
    torch.save(model, model_name)
    with open(preprocessor_name, 'wb') as f:
        dill.dump(preprocessor, f)

def load_model(path, feature_class, tag="best"):
    r"""Loads a model and a preprocessor (the former with PyTorch, the latter with dill).
    """
    model_name, preprocessor_name = names(path, feature_class, tag)
    model = torch.load(model_name)
    with open(preprocessor_name, 'rb') as f:
        preprocessor = dill.load(f)
    return model, preprocessor