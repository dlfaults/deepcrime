import os
import run_deepcrime_properties as dc_props
from deep_crime import mutate as run_deepcrime_tool


def run_automate():

    subject_name = "mnist"
    subject_path = os.path.join('test_models', 'mnist_conv.py')

    dc_props.write_properties(subject_name, subject_path)


    run_deepcrime_tool()

    print("Finished all, exit")


if __name__ == '__main__':
    run_automate()