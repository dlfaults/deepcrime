import copy
import random
import utils.constants as const
import utils.properties as props
from keras.optimizers import Optimizer as KO
from tensorflow.keras.optimizers import Optimizer as TKO

def operator_change_optimisation_function(old_optimiser = None):
    """Unparse the ast tree, save code to py file.

        Keyword arguments:
        tree -- ast tree
        save_path -- the py file where to save the code

        Returns: int (0 - success, -1 otherwise)
    """

    # new_optimiser = random.choice(const.keras_optimisers)#.remove(old_act_func)

    if props.change_optimisation_function["optimisation_function_udp"] is not None:
        new_optimiser = props.change_optimisation_function["optimisation_function_udp"]
    elif props.change_optimisation_function["mutation_target"] is None:
        optimisers = copy.copy(const.keras_optimisers)

        if isinstance(old_optimiser, str):
            if old_optimiser.lower() in optimisers:
                optimisers.remove(old_optimiser.lower())
        elif issubclass(type(old_optimiser), KO) or issubclass(type(old_optimiser), TKO):
            old_optimiser_name = type(old_optimiser).__name__.lower()
            if old_optimiser_name in optimisers:
                optimisers.remove(old_optimiser_name)
        print("Old optimiser:"+ old_optimiser_name)
        new_optimiser = random.choice(optimisers)

        props.change_optimisation_function["mutation_target"] = new_optimiser
    else:
        new_optimiser = props.change_optimisation_function["mutation_target"]

    print("New Optimiser is:" + str(new_optimiser))
    return new_optimiser