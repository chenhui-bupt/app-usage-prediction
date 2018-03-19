import json
import pickle
import os
import re
import hashlib


def uniform_json(jdict):
    """uniform the json dict, make all the keys and values be orderly"""
    if not isinstance(jdict, dict):
        return sorted(list(jdict))  # 使value有序
    keys = sorted(jdict.keys())  # 使键值有序
    unif_dict = dict()
    for key in keys:
        values = uniform_json(jdict[key])  # recursive call
        unif_dict[key] = values
    return unif_dict


def hashlib_json(json_dict):
    string_json = str(json.dumps(uniform_json(json_dict)))
    hashed_json = hashlib.sha256(string_json.encode()).hexdigest()
    return hashed_json




def hash_json_test():
    a = {'u1': ['a1', 'a2'], 'u2': ['a1'], 'u3': ['a2'], 'a1': ['u1', 'u2'], 'a2': ['u1', 'u3']}
    b = {'u2': ['a1'], 'u1': ['a1', 'a2'], 'u3': ['a2'], 'a1': ['u1', 'u2'], 'a2': ['u1', 'u3']}
    c = {'u1': {'a1': ['l1', 'l2'], 'a2': ['l1']}, 'u2': ['a2']}
    d = {'u2': ['a2'], 'u1': {'a2': ['l1'], 'a1': ['l2', 'l1']}}
    print("\n")
    print(uniform_json(a))
    print(uniform_json(b))
    print("\n")
    print(uniform_json(c))
    print(uniform_json(d))
    print("haha\n")
    print(hashlib_json(c))
    print(hashlib_json(d))

# test the hash_json()
# hash_json_test()

filter(lambda x : re.search('graph_id_', x), os.listdir('./cache/'))
os.listdir('./cache/')