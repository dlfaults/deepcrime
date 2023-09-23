import os
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import importlib

import utils.properties as props
import utils.constants as const
import run_deepcrime_properties as dc_props
from deep_crime import mutate as run_deepcrime_tool
from utils.constants import save_paths
from mutation_score import calculate_dc_ms

data = {
    'subject_name': '',
    'subject_path': '',
    'root': os.path.dirname(os.path.abspath(__file__)),
    'mutations': [],
    'mode': 'test'
}

def run_automate():

    data['subject_name'] = 'mnist_pytorch_learning_rate'
    data['subject_path'] = os.path.join('test_models', 'mnist_pytorch.py')
    data['mutations'] = ["change_pytorch_learning_rate"]

    dc_props.write_properties(data)

    shutil.copyfile(os.path.join('utils', 'properties', 'properties_example.py'),
                    os.path.join('utils', 'properties.py'))
    shutil.copyfile(os.path.join('utils', 'properties', 'constants_example.py'),
                    os.path.join('utils', 'constants.py'))

    importlib.reload(props)
    importlib.reload(const)

    print()
    print()
    print("====PROPS DICTIONARY BEFOREHAND======")
    print(props.model_properties)
    print()
    print()
    run_deepcrime_tool()

    if props.MS == 'DC_MS':
        data['mode'] = 'train'
        data['subject_path'] = os.path.join('test_models', 'mnist_pytorch_train.py')
        dc_props.write_properties(data)

        test_results = os.path.join(data['root'], save_paths['mutated'],  data['subject_name'], 'results')
        print(test_results)

        if os.path.isdir(test_results):
            if os.path.isdir(test_results + '_test'):
                shutil.rmtree(test_results + '_test')
            shutil.move(test_results, test_results + '_test')
        else:
            raise Exception()

        # props.model_properties["x_train_len"] = 10000
        print()
        print()
        print("====PROPS DICTIONARY AFTER======")
        print(props.model_properties)
        print()
        print()
        run_deepcrime_tool()

        if os.path.isdir(test_results):
            if os.path.isdir(test_results + '_train'):
                shutil.rmtree(test_results + '_train')
            shutil.move(test_results, test_results + '_train')
        else:
            raise Exception()
        
        train_accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_train")
        accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_test")
        calculate_dc_ms(train_accuracy_dir, accuracy_dir)

    print("Finished all, exit")


if __name__ == '__main__':
    run_automate()