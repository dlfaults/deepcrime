import os
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def write_properties(subject_name, subject_path):
    data = {}

    # General
    data['root'] = os.path.dirname(os.path.abspath(__file__))
    # Subject Related
    data['subject_name'] = subject_name
    data['subject_path'] = subject_path
    data['mutations'] = ["change_activation_function"]

    with open('deep_crime_properties.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_properties():
    with open('deep_crime_properties.json') as data_file:
        data = json.load(data_file)
    return data
