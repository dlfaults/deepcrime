import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import csv
import importlib

import utils.properties as props
import utils.constants as const
import utils.exceptions as e

from utils.mutation_utils import get_accuracy_list_from_scores, update_mutation_properties, load_scores_from_csv
from utils.mutation_utils import concat_params_for_file_name, save_scores_csv, modify_original_model, rename_trained_model
from stats import is_diff_sts
from run_deepcrime_properties import read_properties


scores = []


def execute_mutants(mutants_path, mutations):
    global scores
    mutants = []

    for root, dirs, files in os.walk(mutants_path):
        for file in files:
            if file.endswith(".py"):

                mutants.append([root,file])

    for mutation in mutations:

        my_mutants = [mutant for mutant in mutants if mutation in mutant[1]]

        try:
            mutation_params = getattr(props, mutation)
        except AttributeError:
            print("No attributes found")

        model_params = getattr(props, "model_properties")

        udp = [value for key, value in mutation_params.items() if "udp" in key.lower() and "layer" not in key.lower()]

        if len(udp) > 0:
            udp = udp[0]
        else:
            udp = None

        layer_udp = mutation_params.get("layer_udp", None)


        search_type = mutation_params.get("search_type")

        for mutant in my_mutants:
            if mutation_params.get("layer_mutation", False):
                if layer_udp:
                    if isinstance(layer_udp, list):
                        inds = layer_udp
                    else:
                        inds = [layer_udp]
                else:
                    inds = range(model_params["layers_num"])

                for ind in inds:
                #for ind in range(7, 8):
                    # print("index is:" + str(ind))
                    mutation_params["mutation_target"] = None
                    mutation_params["current_index"] = ind
                    mutation_ind = "_" + str(ind)

                    execute_based_on_search(udp, search_type, mutation, mutant, mutation_params, ind, mutation_ind)
            else:
                execute_based_on_search(udp, search_type, mutation, mutant, mutation_params)


def execute_based_on_search(udp, search_type, mutation, mutant, mutation_params, ind = None, mutation_ind = ''):

    global scores

    try:
        if udp or (search_type is None):
            original_accuracy_list = get_accuracy_list_from_scores(scores)
            mutation_accuracy_list = get_accuracy_list_from_scores(
                execute_mutant(mutant, mutation_params, mutation_ind))

            is_sts, p_value, effect_size = is_diff_sts(original_accuracy_list, mutation_accuracy_list)

            csv_file = os.path.join(mutant[0], "results", "stats", mutation_params['name'] + "_nosearch.csv")

            with open(csv_file, 'a') as f1:
                writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                if ind:
                    writer.writerow([udp, str(p_value), str(effect_size), str(is_sts)])
                else:
                    writer.writerow([udp, str(p_value), str(effect_size), str(is_sts)])
        elif search_type == 'binary':
            print("calling binary search")
            execute_binary_search(mutant, mutation, mutation_params)
        else:
            print("calling exhaustive search")
            execute_exhaustive_search(mutant, mutation, mutation_params, mutation_ind)

    except (e.AddAFMutationError) as err:
        print(err.message + err.expression)


def execute_mutant(mutant_path, mutation_params, mutation_ind = ''):
    scores = []
    params_list = concat_params_for_file_name(mutation_params)

    # raise  Exception()
    trained_mutants_location = os.path.join(os.getcwd(), const.save_paths["trained"])

    try:
        transformed_path = os.path.join(mutant_path[0], mutant_path[1])
        transformed_path = transformed_path.replace(os.path.sep, ".").replace(".py", "")

        m1 = importlib.import_module(transformed_path)

        data = read_properties()
        if data['mode'] in ('train', 'weak'):
            importlib.reload(m1)

        results_file_path = os.path.join(mutant_path[0], "results", mutant_path[1].replace(".py", "") + "_MP" + params_list + mutation_ind + ".csv")

        if not (os.path.isfile(results_file_path)):
            for i in range(mutation_params["runs_number"]):

                mutation_final_name = mutant_path[1].replace(".py", "") + "_MP" + params_list + mutation_ind + "_" + str(i) + ".h5"

                score = m1.main(mutation_final_name)
                scores.append(score)

                path_trained = [trained_mutants_location,
                                props.model_name + "_trained.h5",
                                mutation_final_name]

                rename_trained_model(path_trained)

            if scores:
                save_scores_csv(scores, results_file_path, params_list)
        else:
            scores = load_scores_from_csv(results_file_path)

    except ImportError as err:
        print('Error:', err)
    else:
        a = 1

    return scores

def execute_original_model(model_path, results_path):
    global scores
    scores = []
    modified_model_path = modify_original_model(model_path)

    transformed_path = modified_model_path.replace(os.path.sep, ".").replace(".py", "")

    m1 = importlib.import_module(transformed_path)

    csv_file_path = os.path.join(results_path, props.model_name + ".csv")

    if not(os.path.isfile(csv_file_path)):
        for i in range(const.runs_number_default):
        # for i in range(1):
            path_trained = [os.path.join(os.getcwd(), const.save_paths["trained"]),
                            props.model_name + "_trained.h5",
                            props.model_name + "_original_" + str(i) + ".h5"]

            score = m1.main(path_trained[2])

            scores.append(score)

            rename_trained_model(path_trained)

        save_scores_csv(scores, csv_file_path)
    else:
        print("reading scores from file")
        scores = load_scores_from_csv(csv_file_path)

    return scores


def execute_exhaustive_search(mutant, mutation, my_params, mutation_ind = ''):

    print("Running Exhaustive Search for" + str(mutant))

    original_accuracy_list = get_accuracy_list_from_scores(scores)

    name = my_params['name']
    if name == 'change_optimisation_function':
        for optimiser in const.keras_optimisers:
            print("Changing into optimiser:" + str(optimiser))
            update_mutation_properties(mutation, "optimisation_function_udp", optimiser)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params))
            is_sts, p_value, effect_size = is_diff_sts(original_accuracy_list, mutation_accuracy_list)

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(optimiser), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_activation_function' or name == 'add_activation_function':
        for activation in const.activation_functions:
            print("Changing into activation:" + str(activation))
            update_mutation_properties(mutation, "activation_function_udp", activation)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params, mutation_ind))
            is_sts, p_value, effect_size = is_diff_sts(original_accuracy_list, mutation_accuracy_list)

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(activation), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_loss_function':
        for loss in const.keras_losses:
            print("Changing into loss:" + str(loss))
            update_mutation_properties(mutation, "loss_function_udp", loss)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params))
            is_sts, p_value, effect_size = is_diff_sts(original_accuracy_list, mutation_accuracy_list)

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(loss), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_dropout_rate':
        for dropout in const.dropout_values:
            print("Changing into dropout rate:" + str(dropout))
            update_mutation_properties(mutation, "rate", dropout)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params, mutation_ind))
            is_sts, p_value, effect_size = is_diff_sts(original_accuracy_list, mutation_accuracy_list)

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(dropout), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_batch_size':
        for batch_size in const.batch_sizes:
            print("Changing into batch size:" + str(batch_size))
            update_mutation_properties(mutation, "batch_size", batch_size)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params))
            is_sts, p_value, effect_size = is_diff_sts(original_accuracy_list, mutation_accuracy_list)

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(batch_size), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'change_weights_initialisation':
        for initialiser in const.keras_initialisers:
            print("Changing into initialisation:" + str(initialiser))
            update_mutation_properties(mutation, "weights_initialisation_udp", initialiser)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params, mutation_ind))
            is_sts, p_value, effect_size = is_diff_sts(original_accuracy_list, mutation_accuracy_list)

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(initialiser), str(p_value), str(effect_size), str(is_sts)])
    elif name == 'add_weights_regularisation':
        for regularisation in const.keras_regularisers:
            print("Changing into regularisation:" + str(regularisation))
            update_mutation_properties(mutation, "weights_regularisation_udp", regularisation)
            mutation_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params, mutation_ind))
            is_sts, p_value, effect_size = is_diff_sts(original_accuracy_list, mutation_accuracy_list)

            if len(mutation_accuracy_list) > 0:
                csv_file = os.path.join(mutant[0], "results", "stats", my_params['name'] + "_exssearch.csv")
                with open(csv_file, 'a') as f1:
                    writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
                    writer.writerow([str(regularisation), str(p_value), str(effect_size), str(is_sts)])


def execute_binary_search(mutant, mutation, my_params):
    print("Running Binary Search for" + str(mutant))

    lower_bound = my_params["bs_lower_bound"]
    upper_bound = my_params["bs_upper_bound"]
    precision = my_params["precision"]
    mutant_name = my_params["name"]

    lower_accuracy_list = get_accuracy_list_from_scores(scores)
    update_mutation_properties(mutation, "pct", upper_bound)
    upper_accuracy_list = get_accuracy_list_from_scores(execute_mutant(mutant, my_params))

    is_sts, p_value, effect_size = is_diff_sts(lower_accuracy_list, upper_accuracy_list)

    csv_file = os.path.join(mutant[0], "results", "stats", mutant_name + "_binarysearch.csv")
    with open(csv_file, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow([str(lower_bound), str(upper_bound), '', str(p_value), str(effect_size), str(is_sts)])

    if is_sts:
        print("Binary Search: Upper Bound is Killable")
        search_for_bs_conf(mutant, mutation, my_params, lower_bound, upper_bound, lower_accuracy_list, upper_accuracy_list)
    else:
        print("Binary Search: Upper Bound is Not Killable")


def search_for_bs_conf(mutant, mutation, my_params, lower_bound, upper_bound, lower_accuracy_list, upper_accuracy_list):
    if my_params['bs_rounding_type'] == 'int':
        middle_bound = round((upper_bound + lower_bound) / 2)
    elif my_params['bs_rounding_type'] == 'float3':
        middle_bound = round((upper_bound + lower_bound) / 2, 3)
    elif my_params['bs_rounding_type'] == 'float4':
        middle_bound = round((upper_bound + lower_bound) / 2, 4)
    elif my_params['bs_rounding_type'] == 'float5':
        middle_bound = round((upper_bound + lower_bound) / 2, 5)
    else:
        middle_bound = round((upper_bound + lower_bound) / 2, 2)

    print("middle_bound is:" + str(middle_bound))
    update_mutation_properties(mutation, "pct", middle_bound)
    middle_scores = execute_mutant(mutant, my_params)
    middle_accuracy_list = get_accuracy_list_from_scores(middle_scores)

    is_sts, p_value, effect_size = is_diff_sts(lower_accuracy_list, middle_accuracy_list)
    csv_file = os.path.join(mutant[0], "results", "stats", my_params["name"] + "_binarysearch.csv")
    with open(csv_file, 'a') as f1:
        writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
        writer.writerow([str(lower_bound), str(upper_bound), str(middle_bound), str(p_value), str(effect_size), str(is_sts)])

    if is_sts:
        upper_bound = middle_bound
        upper_accuracy_list = middle_accuracy_list
    else:
        lower_bound = middle_bound
        lower_accuracy_list = middle_accuracy_list

    if abs(upper_bound - lower_bound) <= my_params['precision']:
        if is_sts:
            perfect = middle_bound
            conf_nk = lower_bound
        else:
            perfect = upper_bound
            conf_nk = middle_bound

        csv_file = os.path.join(mutant[0], "results", "stats", my_params["name"] + "_binarysearch.csv")
        with open(csv_file, 'a') as f1:
            writer = csv.writer(f1, delimiter=',', lineterminator='\n', )
            writer.writerow([str(perfect), str(conf_nk)])

        print("Binary Search Configuration is:" + str(perfect))
        return perfect, conf_nk
    else:
        print("Changing interval to: [" + str(lower_bound) + ", " + str(upper_bound) + "]")
        return search_for_bs_conf(mutant, mutation, my_params, lower_bound, upper_bound,
                                  lower_accuracy_list, upper_accuracy_list)


