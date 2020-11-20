from bert_faqclass import _get_config_dict
import json


class ObjectFromDict:
    # constructor
    def __init__(self, dict1: dict):
        self.__dict__.update(dict1)


def dict2obj(dictionary: dict):
    # using json.loads method and passing json.dumps
    # method and custom object hook as arguments
    return json.loads(json.dumps(dictionary), object_hook=ObjectFromDict)


config = dict2obj(_get_config_dict(name="default"))
