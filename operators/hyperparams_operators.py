from tensorflow.keras import backend as K

import utils.properties as props


def operator_change_learning_rate(optimiser):
    """Unparse the ast tree, save code to py file.

        Keyword arguments:
        tree -- ast tree
        save_path -- the py file where to save the code

        Returns: int (0 - success, -1 otherwise)
    """

    if props.change_learning_rate["learning_rate_udp"]:
        new_lr = props.change_learning_rate["learning_rate_udp"]
    else:
        new_lr = props.change_learning_rate["pct"]

    #TODO dopilit "adam"
    with K.name_scope(optimiser.__class__.__name__):
        optimiser.learning_rate = K.variable(new_lr, name='learning_rate')

    return optimiser


# def operator_change_batch_size(batch_size):
#     """
#     Mutate batch size
#
#     Keyword arguments:
#     batch_size -- batch size used in the model
#
#     Returns: int (new batch size)
#     """
#
#     # TODO: something mean with the batch size
#
#     batch_size = 5
#
#     return batch_size
#
#
# def operator_change_epochs(epochs):
#     """ Mutate number of epochs
#
#     Keyword arguments:
#     epochs -- epochs used in the model
#
#     Returns: int (new epochs)
#     """
#
#     # TODO: something mean with the epochs
#
#     epochs = 10
#
#     return epochs
