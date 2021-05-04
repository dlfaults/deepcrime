###
# udp = user_defined_params
# pct = percentage
# lbl = label
###
# Training Data Mutations
###
model_name = ""
model_type = "classification"
statistical_test = "GLM"
MS = "DC_MS"

model_properties = {
    "epochs": 12,
    "batch_size": 128,
    "learning_rate": 1.0,
    "x_train_len": 60000,
    "layers_num": 8,
    "dropout_layers": {3, 6}
}

# Mutation Change label
change_label = {
    "name": 'change_label',
    "change_label_udp": False,
    "change_label_pct": -1,
    "change_label_label": None,
    "runs_number": 10,
    "annotation_params": ["y_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 100,
    "precision": 5,
    "search_type": 'binary',
    "bs_rounding_type": 'float'
}
# change_label_udp = False
# change_label_pct = -1
# change_label_label = None

# Mutation Delete Training Data
delete_training_data = {
    "name": 'delete_td',
    "delete_train_data_udp": False,
    "delete_train_data_pct": -1,
    "runs_number": 10,
    "annotation_params": ["x_train", "y_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 99,
    "precision": 5,
    "search_type": 'binary',
    "bs_rounding_type": 'float'
}
# delete_train_data_udp = False
# delete_train_data_pct = -1

# Unbalance Training Data
unbalance_train_data = {
    "name": 'unbalance_td',
    "unbalance_train_data_udp": False,
    "unbalance_train_data_pct": -1,
    "runs_number": 10,
    "annotation_params": ["x_train", "y_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 100,
    "precision": 5,
    "search_type": 'binary',
    "bs_rounding_type": 'float'
}

make_output_classes_overlap = {
    "name": 'output_classes_overlap',
    "make_output_classes_overlap_udp": False,
    "make_output_classes_overlap_pct": -1,
    "runs_number": 10,
    "annotation_params": ["x_train", "y_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 100,
    "precision": 5,
    "search_type": 'binary',
    "bs_rounding_type": 'float'
}

# Add Noise to Training Data
add_noise = {
    "name": 'add_noise',
    "add_noise_udp": False,
    "add_noise_pct": -1,
    "runs_number": 10,
    "annotation_params": ["x_train"],
    "bs_lower_bound": 0,
    "bs_upper_bound": 100,
    "search_type": 'binary',
     "precision": 5,
     "bs_rounding_type": 'float'
}

change_epochs = {
    "name": 'change_epochs',
    "change_epochs_size": False,
    "pct": 12,
    "bs_lower_bound": 12,
    "bs_upper_bound": 1,
    "bs_rounding_type": 'int',
    "annotation_params": [],
    "search_type": 'binary',
    "precision": 1,
    "runs_number": 10,
}

change_batch_size = {
    "name": 'change_batch_size',
    "runs_number": 10,
    "change_batch_size_udp": False,
    "batch_size": -1,
    "annotation_params": [],
    "search_type": 'exhaustive',
    "applicable": True
}

change_learning_rate = {
    "name": 'change_learning_rate',
    "learning_rate_udp": False,
    "pct": -1,
    "bs_lower_bound": 1.0,
    "bs_upper_bound": 0.001,
    "annotation_params": [],
    "search_type": 'binary',
    "runs_number": 10,
    "precision": 0.01,
    "bs_rounding_type": 'float3'
}

disable_batching = {
    "name": 'disable_batching',
    "train_size": 60000,
    "annotation_params": [],
    "search_type": None,
    "runs_number": 10,
    "applicable": True
}

change_activation_function = {
    "name": 'change_activation_function',
    "activation_function_udp": False,
    "layer_udp": 6,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": 'exhaustive'
}

remove_activation_function = {
    "name": 'remove_activation_function',
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": None
}

add_activation_function = {
    "name": 'add_activation_function',
    "activation_function_udp": None,
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": 'exhaustive'
}

change_weights_initialisation = {
    "name": 'change_weights_initialisation',
    "weights_initialisation_udp": None,
    "layer_udp": 0,
    "annotation_params": [],
    "runs_number": 10,
    "current_index": 0,
    "layer_mutation": True,
    "search_type": 'exhaustive'
}

change_optimisation_function = {
    "optimisation_function_udp": "sgd",
    "annotation_params": [],
    "mutation_target": None,
    "runs_number": 10,
    "layer_mutation": False,
    "search_type": None,
    "name": 'change_optimisation_function',
}

remove_validation_set = {
    "name": 'remove_validation_set',
    "runs_number": 10,
    "annotation_params": [],
    "search_type": None
}

change_gradient_clip = {
    "change_gradient_clip_udp": False,
    "clipnorm": -1,
    "bs_lower_bound": 0,
    "bs_upper_bound": 0,
    "clipvalue": 0.5,
    # "bs_lower_bound1": 0,
    # "bs_upper_bound1": 0,
    "annotation_params": [],
    "search_type": None
}

add_bias = {
    "name": 'add_bias',
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": None
}

remove_bias = {
    "name": 'remove_bias',
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": None
}

change_loss_function = {
    "name": 'change_loss_function',
    "loss_function_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "mutation_target": None,
    "search_type": 'exhaustive',
    "layer_mutation": False
}

change_dropout_rate = {
    "name": 'change_dropout_rate',
    "layer_udp": [3, 6],
    "runs_number": 10,
    "dropout_rate_udp": False,
    "annotation_params": [],
    "rate": 0,
    "current_index": 0,
    "layer_mutation": True,
    "search_type": 'exhaustive'
}

add_weights_regularisation = {
    "name": 'add_weights_regularisation',
    "weights_regularisation_udp": None,
    "layer_udp": 0,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": 'exhaustive'
}

change_weights_regularisation = {
    "name": 'change_weights_regularisation',
    "weights_regularisation_udp": None,
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "mutation_target": None,
    "search_type": None
}

remove_weights_regularisation = {
    "name": 'remove_weights_regularisation',
    "layer_udp": None,
    "runs_number": 10,
    "annotation_params": [],
    "layer_mutation": True,
    "current_index": 0,
    "search_type": 'exhaustive'
}

change_earlystopping_patience = {
    "name": "change_patience",
    "runs_number": 10,
    "patience_udp": None,
    "annotation_params": [],
    "layer_mutation": False,
    "bs_lower_bound": 10,
    "bs_upper_bound": 1,
    "pct": 1,
    "search_type": 'binary'
}
