import json
import os

settings = {
    'audio': {
        'subject_name': 'audio',
        'subject_path': os.path.join('test_models', 'audio_model_mod.py'),
        'mutations': ["change_label", "delete_training_data", "unbalance_train_data", "make_output_classes_overlap",
                      "change_batch_size", "change_learning_rate", "change_epochs", "change_activation_function",
                      "remove_activation_function", "add_activation_function","add_weights_regularisation",
                      "change_weights_initialisation", "remove_bias", "change_loss_function",
                      "change_optimisation_function", "remove_validation_set"]
    },

    'mnist': {
        'subject_name': 'mnist',
        'subject_path': os.path.join('test_models', 'mnist_conv.py'),
        'mutations': ["change_label", "delete_training_data", "unbalance_train_data", "add_noise",
                        "make_output_classes_overlap", "change_batch_size", "change_learning_rate", "change_epochs",
                        "disable_batching", "change_activation_function", "remove_activation_function",
                        "add_weights_regularisation", "change_dropout_rate", "change_weights_initialisation",
                        "remove_bias", "change_loss_function", "change_optimisation_function", "remove_validation_set"]
    },

    'lenet': {
        'subject_name': 'lenet',
        'subject_path': os.path.join('test_models', 'lenet.py'),
        'mutations': ["remove_validation_set", "change_optimisation_function", "change_loss_function",
                        "remove_activation_function", "remove_bias", "add_weights_regularisation",
                        "add_activation_function", "change_activation_function", "change_weights_initialisation",
                        "change_epochs", "change_batch_size", "change_learning_rate", "delete_training_data",
                        "add_noise", "unbalance_train_data", "make_output_classes_overlap", "change_label",]


    },

    'udacity': {
        'subject_name': 'udacity',
        'subject_path': os.path.join('test_models', 'train_self_driving_car.py'),
        'mutations': ["change_label", "delete_training_data", "unbalance_train_data", "make_output_classes_overlap",
                      "change_learning_rate", "change_epochs", "change_activation_function",
                      "remove_activation_function", "add_weights_regularisation", "change_dropout_rate",
                      "change_weights_initialisation", "remove_bias", "change_loss_function",
                      "change_optimisation_function", "remove_validation_set",]


    },

    'movie': {
        'subject_name': 'movie',
        'subject_path': os.path.join('test_models', 'movie_recomm_mod.py'),
        'mutations': ["change_label", "delete_training_data", "unbalance_train_data", "make_output_classes_overlap",
                      "change_batch_size", "change_learning_rate", "change_epochs", "disable_batching",
                      "change_loss_function", "change_optimisation_function", "remove_validation_set"]
    }
}


def write_subject_settings():
    global settings
    with open('subject_settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)


def read_subject_settings(subject):
    with open('subject_settings.json') as data_file:
        data = json.load(data_file)

    settings = data.get(subject, None)
    return settings


if __name__ == '__main__':
    write_subject_settings()