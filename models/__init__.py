from models.linearnd import LinearND
from models.basicnetwork import BasicNetwork
from models.progressivenetwork import ProgressiveNetwork

name_to_model = {
    "BasicNetwork": BasicNetwork,
    "ProgressiveNetwork": ProgressiveNetwork
}

model_to_name = {value: key for key, value in name_to_model.items()}