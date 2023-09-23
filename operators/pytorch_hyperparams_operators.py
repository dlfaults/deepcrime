import utils.constants as const
import utils.properties as props


def operator_change_pytorch_batch_size(old_batch):
    if props.change_pytorch_batch_size["batch_size"] != -1:
        # IN EXHAUSTIVE SEARCH, IT GOES INTO HERE - a value for udp is set
        new_batch_size = props.change_pytorch_batch_size["batch_size"]
    else:
        print("ERROR WHEN PERFORMING CHANGE PYTORCH LOSS FUNCTION, setting new loss function to None")
        new_batch_size = None

    return new_batch_size


def operator_change_pytorch_epochs(old_epochs):
    new_epochs = props.change_pytorch_epochs['pct']
    return new_epochs


def operator_change_pytorch_learning_rate(old_lr):
    new_lr = props.change_pytorch_learning_rate['pct']
    return new_lr


def operator_disable_pytorch_batching(old_batch):
    new_batch_size = props.model_properties["x_train_len"]
    print("=====NEW BATCH SIZE IS: " + str(new_batch_size) + "========")
    return new_batch_size
