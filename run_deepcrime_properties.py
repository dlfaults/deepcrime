import os
import json


def write_properties(data):
    with open('deep_crime_properties.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_properties():
    with open('deep_crime_properties.json') as data_file:
        data = json.load(data_file)
    return data
