import copy
import random
import utils.constants as const
import utils.properties as props


def operator_change_pytorch_optimiser(old_optimiser=None):
    """Unparse the ast tree, save code to py file.

        Keyword arguments:
        tree -- ast tree
        save_path -- the py file where to save the code

        Returns: int (0 - success, -1 otherwise)
    """

    if props.change_pytorch_optimisation_function["optimisation_function_udp"] is not None:
        new_optimiser = props.change_pytorch_optimisation_function["optimisation_function_udp"]
    elif props.change_pytorch_optimisation_function["mutation_target"] is None:
        optimisers = copy.copy(const.pytorch_optimisers)

        if isinstance(old_optimiser, str):
            if old_optimiser.lower() in optimisers:
                optimisers.remove(old_optimiser.lower())
        new_optimiser = random.choice(optimisers)

        props.change_pytorch_optimisation_function["mutation_target"] = new_optimiser
    else:
        new_optimiser = props.change_pytorch_optimisation_function["mutation_target"]

    print("New Optimiser is:" + str(new_optimiser))
    return const.pytorch_optimisers.get(new_optimiser)
