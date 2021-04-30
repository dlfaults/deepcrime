import copy
import random
import utils.constants as const
import utils.properties as props
from tensorflow.keras.losses import Loss as TKLSS
from keras.losses import Loss as KLSS


def operator_change_loss_function(old_loss):
    if props.change_loss_function["loss_function_udp"] is not None:
        new_loss_func = props.change_loss_function["loss_function_udp"]

    elif props.change_loss_function["mutation_target"] is None:
        loss_functions = copy(const.keras_losses)

        if isinstance(old_loss, str):
            loss_functions.remove(old_loss)
        elif issubclass(type(old_loss), TKLSS) or issubclass(type(old_loss), KLSS):
            old_loss_name = old_loss.name
            if old_loss_name in loss_functions:
                loss_functions.remove(old_loss_name)
        else:
            print("Custom loss detected")

        new_loss_func = random.choice(loss_functions)
        props.change_loss_function["mutation_target"] = new_loss_func
        print("New Loss Function is:" + str(new_loss_func))
    else:
        new_loss_func = props.change_loss_function["mutation_target"]

    return new_loss_func
