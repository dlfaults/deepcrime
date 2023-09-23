import copy
import random
import utils.constants as const
import utils.properties as props
# from tensorflow.keras.losses import Loss as TKLSS
# from keras.losses import Loss as KLSS
import torch.nn.functional as F


def operator_change_pytorch_loss_function(old_loss):
    if props.change_pytorch_loss_function["loss_function_udp"] is not None:
        # IN EXHAUSTIVE SEARCH, IT GOES INTO HERE - a value for udp is set
        new_loss_func = props.change_pytorch_loss_function["loss_function_udp"]
    else:
        print("ERROR WHEN PERFORMING CHANGE PYTORCH LOSS FUNCTION, setting new loss function to None")
        new_loss_func = None
    # elif props.change_pytorch_loss_function["mutation_target"] is None:
    # loss_functions = copy(const.pytorch_losses)

    # if isinstance(old_loss, str):
    # loss_functions.remove(old_loss)
    # elif issubclass(type(old_loss), TKLSS) or issubclass(type(old_loss), KLSS):
    #   old_loss_name = old_loss.name
    #   if old_loss_name in loss_functions:
    #   loss_functions.remove(old_loss_name)
    # else:
    #   print("Custom loss detected")

    # new_loss_func = random.choice(loss_functions)
    # props.change_pytorch_loss_function["mutation_target"] = new_loss_func
    # print("New Loss Function is:" + str(new_loss_func))
    # else:
    #     new_loss_func = props.change_pytorch_loss_function["mutation_target"]

    # convert the new loss function we are getting, into the actual torch loss function:

    # THINK ABOUT MORE EFFICIENT WAY OF DOING THIS, maybe turn pytorch_losses in constants.py into dictionary
    # with e.g. "nll_loss" : F.nll_loss
    # and here, we just return the value of the key of the loss function we are using
    # if new_loss_func == "nll_loss":
    #    return F.nll_loss
    # elif new_loss_func == "cross_entropy":
    #     return F.cross_entropy
    # else:
    #     return F.multi_margin_loss

    return const.pytorch_losses.get(new_loss_func)
