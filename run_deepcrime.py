import os
import shutil

import run_deepcrime_properties as dc_props
from deep_crime import mutate as run_deepcrime_tool
from utils.constants import save_paths
from utils.properties import MS

data = {
    'subject_name': '',
    'subject_path': '',
    'root': os.path.dirname(os.path.abspath(__file__)),
    'mutations': [],
    'mode': 'test'
}

def run_automate():

    data['subject_name'] = 'mnist'
    data['subject_path'] = os.path.join('test_models', 'mnist_conv.py')
    data['mutations'] = ["change_optimisation_function"]

    dc_props.write_properties(data)

    run_deepcrime_tool()

    if MS == 'DC_MS':
        data['mode'] = 'train'
        data['subject_path'] = os.path.join('test_models', 'mnist_conv_train.py')
        dc_props.write_properties(data)

        test_results = os.path.join(data['root'], save_paths['mutated'],  data['subject_name'], 'results')
        print(test_results)

        if os.path.isdir(test_results):
            shutil.move(test_results, test_results + '_test')
        else:
            raise Exception()


        run_deepcrime_tool()

        if os.path.isdir(test_results):
            shutil.move(test_results, test_results + '_train')
        else:
            raise Exception()

        import mutation_score

    print("Finished all, exit")


if __name__ == '__main__':
    run_automate()