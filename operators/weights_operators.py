import random
import copy

import utils.constants as const
import utils.properties as props
import utils.exceptions as e
from utils import mutation_utils as mu


def operator_change_weights_initialisation(model):
    """Unparse the ast tree, save code to py file.

        Keyword arguments:
        tree -- ast tree
        save_path -- the py file where to save the code

        Returns: int (0 - success, -1 otherwise)
    """

    if not model:
        print("raise,log we have probllems")

    current_index = props.change_weights_initialisation["current_index"]

    tmp = model.get_config()
    
    print("Changing Weight initialisation of layer" + str(current_index))

    if tmp['layers'][current_index]['config'].get('kernel_initializer'):

        if props.change_weights_initialisation["weights_initialisation_udp"] is not None:
            new_init = props.change_weights_initialisation["weights_initialisation_udp"]
        elif props.change_weights_initialisation["mutation_target"] is None:

            initialisers = copy.copy(const.keras_initialisers)
            initialisers_formatted = [element.replace('_','') for element in initialisers]

            old_init = tmp['layers'][current_index]['config']['kernel_initializer'].get('class_name')
            old_init_formatted = old_init.lower().replace('_', '')

            if old_init_formatted in initialisers_formatted:
                idx = initialisers_formatted.index(old_init_formatted)
                del initialisers[idx]
            elif old_init_formatted == 'variancescaling':
                vs_configs = const.keras_vs_initialisers_config
                for vs in vs_configs:
                    if tmp['layers'][current_index]['config']['kernel_initializer']['config'].get('scale') == vs[0]\
                        and tmp['layers'][current_index]['config']['kernel_initializer']['config'].get('mode') == vs[1]\
                        and tmp['layers'][current_index]['config']['kernel_initializer']['config'].get('distribution') == vs[2]:
                        old_init = vs[3]
                        initialisers.remove(old_init)

            if len(model.layers[current_index].weights[0].shape) > 2:
                initialisers.remove("identity")

            new_init = random.choice(initialisers)

            print(tmp['layers'][current_index]['config']['kernel_initializer'])
            print(old_init)

            props.change_weights_initialisation["mutation_target"] = new_init
        else:
            new_init = props.change_weights_initialisation["mutation_target"]

        tmp['layers'][current_index]['config']['kernel_initializer'] = new_init
    else:
        raise e.AddAFMutationError(str(current_index), "Not possible to apply the add activation function mutation to layer ")

    model = mu.model_from_config(model, tmp)

    return model


def operator_change_weights_regularisation(model):
    """Unparse the ast tree, save code to py file.

           Keyword arguments:
           tree -- ast tree
           save_path -- the py file where to save the code

           Returns: int (0 - success, -1 otherwise)
       """
    if not model:
        print("raise,log we have probllems")

    current_index = props.change_weights_regularisation["current_index"]

    tmp = model.get_config()

    regularisers = copy.copy(const.keras_regularisers)

    print("Changing Regulariser of layer" + str(current_index))

    if tmp['layers'][current_index]['config'].get('kernel_regularizer'):
        if props.change_weights_regularisation["weights_regularisation_udp"] is not None:
            new_regulariser = props.change_weights_regularisation["weights_regularisation_udp"]
        elif props.change_weights_regularisation["mutation_target"] is None:

            if tmp['layers'][current_index]['config']['kernel_regularizer']['config'].get('l1') in (0.0, '0.0', '0'):
                old_regulariser = 'l2'
            elif tmp['layers'][current_index]['config']['kernel_regularizer']['config'].get('l2') in (0.0, '0.0', '0'):
                old_regulariser = 'l1'
            else:
                old_regulariser = 'l1_l2'

            if old_regulariser in regularisers:
                regularisers.remove(old_regulariser)

            new_regulariser = random.choice(regularisers)
            props.change_weights_regularisation["mutation_target"] = new_regulariser
        else:
            new_regulariser = props.change_weights_regularisation["mutation_target"]


        tmp['layers'][current_index]['config']['kernel_regularizer'] = new_regulariser
    else:
        raise e.AddAFMutationError(str(current_index),
                                   "Not possible to apply the change weights regularisation mutation to the layer ")

    model = mu.model_from_config(model, tmp)

    return model


def operator_add_weights_regularisation(model):
    """Unparse the ast tree, save code to py file.

           Keyword arguments:
           tree -- ast tree
           save_path -- the py file where to save the code

           Returns: int (0 - success, -1 otherwise)
       """
    if not model:
        print("raise,log we have probllems")

    current_index = props.add_weights_regularisation["current_index"]

    tmp = model.get_config()


    print("Add Regulariser to a layer" + str(current_index))

    if "kernel_regularizer" in tmp['layers'][current_index]['config'] and \
            tmp['layers'][current_index]['config'].get('kernel_regularizer') is None:
        if props.add_weights_regularisation["weights_regularisation_udp"] is not None:
            new_regulariser = props.add_weights_regularisation["weights_regularisation_udp"]
        elif props.add_weights_regularisation["mutation_target"] is None:
            regularisers = copy.copy(const.keras_regularisers)
            new_regulariser = random.choice(regularisers)
            props.add_weights_regularisation["mutation_target"] = new_regulariser
        else:
            new_regulariser = props.add_weights_regularisation["mutation_target"]

        print("____________________________________")
        print("Current Index: " + str(current_index))
        print("New Reg:" + new_regulariser)

        tmp['layers'][current_index]['config']['kernel_regularizer'] = new_regulariser
    else:
        raise e.AddAFMutationError(str(current_index),
                                   "Not possible to apply the add weights regularisation mutation to the layer ")

    model = mu.model_from_config(model, tmp)

    return model


def operator_remove_weights_regularisation(model):
    """Unparse the ast tree, save code to py file.

           Keyword arguments:
           tree -- ast tree
           save_path -- the py file where to save the code

           Returns: int (0 - success, -1 otherwise)
       """
    if not model:
        print("raise,log we have probllems")

    current_index = props.remove_weights_regularisation["current_index"]

    tmp = model.get_config()


    print("Remove Regulariser to a layer" + str(current_index))

    if tmp['layers'][current_index]['config'].get('kernel_regularizer') is not None:
        print("Regulariser is:" + tmp['layers'][current_index]['config'].get('kernel_regularizer'))
        tmp['layers'][current_index]['config']['kernel_regularizer'] = None
    else:
        raise e.AddAFMutationError(str(current_index),
                                   "Not possible to apply the add weights regularisation mutation to the layer ")

    model = mu.model_from_config(model, tmp)

    return model