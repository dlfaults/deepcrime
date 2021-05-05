import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
import importlib

import utils.properties as props
import utils.constants as const
import run_deepcrime_properties as dc_props
from deep_crime import mutate as run_deepcrime_tool
from utils.constants import save_paths
from utils.write_settings import write_subject_settings, read_subject_settings
from mutation_score import calculate_dc_ms


data = {
    'subject_name': '',
    'subject_path': '',
    'root': os.path.dirname(os.path.abspath(__file__)),
    'mutations': [],
    'mode': 'test'
}

def run_automate():
    write_subject_settings()

    subjects = ['mnist', 'lenet', 'audio', 'udacity', 'movie']
    # subjects = ['mnist']
    # subjects = ['lenet']
    # subjects = ['audio']
    # subjects = ['udacity']
    # subjects = ['movie']

    for subject in subjects:
        # Get subject details
        data = read_subject_settings(subject)
        data['root'] = os.path.dirname(os.path.abspath(__file__))
        data['mode'] = 'test'

        dc_props.write_properties(data)

        # Copy subject properties and constants
        shutil.copyfile(os.path.join('utils', 'properties', 'properties_' + subject + ".py"),
                        os.path.join('utils', 'properties.py'))
        shutil.copyfile(os.path.join('utils', 'properties', 'constants_' + subject + ".py"),
                        os.path.join('utils', 'constants.py'))

        importlib.reload(props)
        importlib.reload(const)

        # Run DeepCrime for the evaluation on the original strong test set
        run_deepcrime_tool()

        # If we do calculate DeepCrime mutation score for the subject (not the case for MovieRecommender subject)
        if props.MS == 'DC_MS':
            # Run DeepCrime for the evaluation on the train set
            data['mode'] = 'train'
            data['subject_path'] = data['subject_path'].replace('.py', '_train.py')
            dc_props.write_properties(data)

            test_results = os.path.join(data['root'], save_paths['mutated'],  data['subject_name'], 'results')
            print(test_results)

            if os.path.isdir(test_results):
                if os.path.isdir(test_results + '_test'):
                    shutil.rmtree(test_results + '_test')
                shutil.move(test_results, test_results + '_test')
            else:
                raise Exception()

            run_deepcrime_tool()

            if os.path.isdir(test_results):
                if os.path.isdir(test_results + '_train'):
                    shutil.rmtree(test_results + '_train')
                shutil.move(test_results, test_results + '_train')
            else:
                raise Exception()

            # Calculate the DeepCrime mutation score for the original test set
            train_accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_train")
            accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_test")
            calculate_dc_ms(train_accuracy_dir, accuracy_dir)

            # Run DeepCrime for the evaluation on the weak test set
            data['mode'] = 'weak'
            data['subject_path'] = data['subject_path'].replace('_train.py', '_weak.py')
            dc_props.write_properties(data)

            run_deepcrime_tool()

            if os.path.isdir(test_results):
                if os.path.isdir(test_results + '_weak'):
                    shutil.rmtree(test_results + '_weak')
                shutil.move(test_results, test_results + '_weak')
            else:
                raise Exception()

            # Calculate the DeepCrime mutation score for the weak test set
            train_accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_train")
            accuracy_dir = os.path.join("mutated_models", data['subject_name'], "results_weak")
            calculate_dc_ms(train_accuracy_dir, accuracy_dir)

    print("Finished all, exit")


if __name__ == '__main__':
    run_automate()