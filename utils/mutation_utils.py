import ast
import shutil
import string
import random
import csv
import os

from io import StringIO

import tensorflow as tf
import keras
# import keras.engine.sequential
from keras import Sequential as KS
from tensorflow.keras import Sequential as TKS
from keras.models import Model as KM
from tensorflow.keras import Model as TKM

from keras.engine.sequential import Sequential as KES
from keras.engine.training import Model as KEM
from tensorflow.python.keras.engine.sequential import Sequential as TKES
from tensorflow.python.keras.engine.training import Model as TKEM

import mutations
import utils.properties as props
import utils.constants as const
from utils.unparse import Unparser
from utils.logger_setup import setup_logger


logger = setup_logger(__name__)
########################################################################################################################
###################################### PREP&FINISH FUNCTIONS ###########################################################

def generate_import_nodes(mutation_types):
    """Generate an import node based on the mutation_type

        Keyword arguments:
        types -- list of types of mutations to be done on a model in question

        Returns: list of ast import nodes
    """

    # TODO: get the dict of (mutation type, mutation lib name)

    import_nodes = []

    # for type in mutation_types:
    #     # import_nodes.append(ast.Import(names=[ast.alias(name=const.mutation_imports[type], asname=None)]))
    #     import_nodes.append(
    #         ast.ImportFrom(module=const.operator_lib, names=[
    #             ast.alias(name=const.mutation_imports[type], asname=None),
    #         ], level=0))

    for imp in const.operator_lib:
        # import_nodes.append(ast.Import(names=[ast.alias(name=const.mutation_imports[type], asname=None)]))
        import_nodes.append(
            ast.ImportFrom(module=const.operator_mod, names=[
                ast.alias(name=imp, asname=None),
            ], level=0))

    # import_nodes.append(
    #     ast.ImportFrom(module="operators", names=[
    #         ast.alias(name="training_process_operators", asname=None),
    #     ], level=0))
    #
    # import_nodes.append(
    #     ast.ImportFrom(module="operators", names=[
    #         ast.alias(name="activation_function_operators", asname=None),
    #     ], level=0))

    import_nodes.append(
        ast.ImportFrom(module="utils", names=[
            ast.alias(name="mutation_utils", asname=None),
        ], level=0))

    import_nodes.append(
        ast.ImportFrom(module="utils", names=[
            ast.alias(name="properties", asname=None),
        ], level=0))

    import_nodes.append(
        ast.ImportFrom(module="keras", names=[
            ast.alias(name="optimizers", asname=None),
        ], level=0))

    return import_nodes


def prepare_model(file_path, save_path_prepared, save_path_trained, mutation_types):
    """Convert a file size to human-readable form.

    Keyword arguments:
    file_path -- path to the file containing the model
    save_path_prepared -- path to the file where to save the changed code
    save_path_trained -- path where to save the trained models
    mutation_types -- types of mutations to be added

    Returns: ...
    """

    # num_ins_imps = 1
    imports_inserted = False

    with open(file_path, "r") as source:
        tree = ast.parse(source.read())
        # print(astpp.dump(tree))
    # add checks and error handling

    for node in ast.walk(tree):
        if hasattr(node, 'body') and isinstance(node.body, list):
            for ind, x in enumerate(node.body):
                if not imports_inserted:
                    if is_import(x):
                        import_nodes = generate_import_nodes(mutation_types)

                        for i, n in enumerate(import_nodes):
                            node.body.insert(ind + i + 1, n)

                        imports_inserted = True

                # if is_training_call(x):
                #     model_save_node = ast.Expr(value=ast.Call(
                #         func=ast.Attribute(value=ast.Name(id='model', ctx=ast.Load()), attr='save',
                #                            ctx=ast.Load()), args=[ast.Str(s=save_path_trained), ],
                #         keywords=[]))
                #
                #     node.body.insert(ind + 1, model_save_node)
                    # break

                # if is_specific_call(x, 'evaluate'):
                #     score_name = x.targets[0].id
                #     model_eval_save_node = ast.Expr(value=ast.Call(
                #         func=ast.Attribute(value=ast.Name(id='mutation_utils', ctx=ast.Load()), attr='save_scores',
                #                            ctx=ast.Load()), args=[ast.Name(id=score_name, ctx=ast.Load()),
                #                                                   ast.Str(s=save_path_trained), ],
                #         keywords=[]))
                #
                #     node.body.insert(ind + 1, model_eval_save_node)

    # fix the missing locations in ast tree
    ast.fix_missing_locations(tree)

    # unparse the ast tree, write the resulting code into py file
    unparse_tree(tree, save_path_prepared)

    # return save_path


def unparse_tree(tree, save_path):
    """Unparse the ast tree, save code to py file.

        Keyword arguments:
        tree -- ast tree
        save_path -- the py file where to save the code

        Returns: int (0 - success, -1 otherwise)
    """

    buf = StringIO()

    try:
        Unparser(tree, buf)
        buf.seek(0)

        with open(save_path, 'w') as fd:
            buf.seek(0)
            shutil.copyfileobj(buf, fd)
            fd.close()
    except Exception as e:
        logger.error("Unable to unparse the tree: " + str(e))
        raise


def create_mutation(mut):
    """ Script that renames the file with trained model

        Keyword arguments:
        file_path -- path to the file
        ... params needed to constuct new name

        Returns: ...
    """

    MutClass = getattr(mutations, const.mutation_class_map[mut])

    if MutClass is None:
        logger.error("Has not found a class to create a Mutation.")
        raise LookupError("Has not found a class to create a Mutation")

    mutation = MutClass()

    return mutation


def modify_original_model(model_path):
    with open(model_path, "r") as source:
        tree = ast.parse(source.read())

    # params = {}
    keywords = []

    imports_inserted = False

    save_path_prepared = model_path.replace(".py", "_orig.py")

    for node in ast.walk(tree):
        if hasattr(node, 'body') and isinstance(node.body, list):
            for ind, x in enumerate(node.body):
                if not imports_inserted:
                    if is_import(x):
                        import_nodes = []
                        import_nodes.append(
                            ast.ImportFrom(module="utils", names=[
                                ast.alias(name="mutation_utils", asname=None),
                            ], level=0))

                        for i, n in enumerate(import_nodes):
                            node.body.insert(ind + i + 1, n)

                        imports_inserted = True

                if is_specific_call(x, 'compile'):
                    model_name = x.value.func.value.id
                    lr_save_node = ast.Expr(value=ast.Call(
                                func=ast.Attribute(value=ast.Name(id='mutation_utils', ctx=ast.Load()), attr='save_original_model_params',
                                                   ctx=ast.Load()), args=[ast.Name(id=model_name, ctx=ast.Load()), ],
                                keywords=[]))
                    node.body.insert(ind + 1, lr_save_node)

                if is_specific_call(x, 'fit'):

                    if hasattr(x.value, 'args') and len(x.value.args) > 0:
                        if isinstance(x.value.args[0], ast.List):
                            x_train = ast.Name(id=x.value.args[0].elts[0].id, ctx=ast.Load())
                        else:
                            x_train = ast.Name(id=x.value.args[0].id, ctx=ast.Load())

                        keywords.append(
                            ast.keyword(arg="x", value=x_train))

                    # else:
                    #     print("we have a problem here")

                    if hasattr(x.value, 'keywords') and len(x.value.keywords) > 0:
                        for k in x.value.keywords:
                            if k.arg in ("batch_size", "epochs", "x"):
                                keywords.append(k)
                    # else:
                    #     print("we have a problem here")

                    param_save_node = ast.Expr(value=ast.Call(
                        func=ast.Attribute(value=ast.Name(id='mutation_utils', ctx=ast.Load()), attr='save_original_fit_params',
                                           ctx=ast.Load()), args=[], keywords=keywords))

                    node.body.insert(ind, param_save_node)
                    break
    # fix the missing locations in ast tree
    ast.fix_missing_locations(tree)

    # unparse the ast tree, write the resulting code into py file
    unparse_tree(tree, save_path_prepared)

    return save_path_prepared


def save_original_model_params(model):
    dropout_layers = {}

    for attr, value in model.__dict__.items():
        if attr == "optimizer":
            lr = model.__dict__.get('optimizer').__dict__.get('learning_rate')
            lr_value = tf.keras.backend.get_value(lr)
            props.model_properties["learning_rate"] = lr_value

    if model.layers:
        props.model_properties["layers_num"] = len(model.layers)

        for ind, layer in enumerate(model.layers):
            if isinstance(layer, keras.layers.core.Dropout):
                dropout_layers[ind] = [layer.name, layer.rate]

        props.model_properties["dropout_layers"] = dropout_layers

    else:
        print("model has no layers")

def save_original_fit_params(x = None, epochs = None, batch_size = None):
    # if x.any():
    if x is not None:
        try:
            props.model_properties["x_train_len"] = len(x)
        except:
            props.disable_batching["applicable"] = False
            props.change_batch_size["applicable"] = False
    else:
        props.disable_batching["applicable"] = False
        props.change_batch_size["applicable"] = False

    props.model_properties["epochs"] = epochs
    props.model_properties["batch_size"] = batch_size


########################################################################################################################
###################################### CHECK FUNCTIONS #################################################################

def check_for_annotation(elem, annotation_list):
    """Check if the given element corresponds to an annotation

        Keyword arguments:
        elem -- part of ast node
        annot_type -- type of the annotation: x_train, y_train

        Returns: boolean
    """
    if (is_annotated_node(elem)):
        target = elem.target.id
        annotation = elem.annotation.s

        if annotation in annotation_list:
            annotation_list.update({annotation: target})

def is_annotated_node(elem):
    """Check if the given node is an annotation

        Keyword arguments:
        elem -- part of ast node
        annot_type -- type of the annotation: x_train, y_train

        Returns: boolean
    """
    result = False

    if isinstance(elem, ast.AnnAssign):
        result = True

    return result

def is_specific_call(elem, call_type):
    """Check if the given element corresponds to a specific method call

        Keyword arguments:
        elem -- part of ast node
        call_type -- type of the call: fit, evaluate, add

        Returns: boolean
    """

    is_scall = False

    if (isinstance(elem, ast.Assign)
        and isinstance(elem.value, ast.Call) \
        and isinstance(elem.value.func, ast.Attribute) \
        and elem.value.func.attr == call_type) \
            or (isinstance(elem, ast.Expr)
                and isinstance(elem.value, ast.Call)
                and hasattr(elem.value.func, 'attr')
                and elem.value.func.attr == call_type):
        is_scall = True

    return is_scall


def is_training_call(elem):
    """Check if the given node corresponds to the call to fit/fit_generator

        Keyword arguments:
        elem - ast node

        Returns: boolean
    """

    is_call = False

    if (isinstance(elem, ast.Assign)
        and isinstance(elem.value, ast.Call) \
        and isinstance(elem.value.func, ast.Attribute) \
        # and elem.value.func.attr == 'fit') \
        and elem.value.func.attr in ('fit', 'fit_generator')) \
            or (isinstance(elem, ast.Expr)
                and isinstance(elem.value, ast.Call)
                and hasattr(elem.value.func, 'attr')
                # and elem.value.func.attr == 'fit'):
                and elem.value.func.attr in ('fit', 'fit_generator')):
        is_call = True

    return is_call


def is_activation_def_by_kwd(elem):
    # Type K
    res = False
    type = None
    pos = None

    if hasattr(elem.value, 'args') and len(elem.value.args) > 0 \
            and isinstance(elem.value.args[0], ast.Call) \
            and hasattr(elem.value.args[0], 'keywords'):
        for ptn, kwd in enumerate(elem.value.args[0].keywords):
            if kwd.arg == 'activation':
                res = True
                type = "K"
                pos = ptn

    return res, type, pos


def is_activation_def_by_layer(elem):
    # Type L
    res = False
    type = None
    pos = None

    if isinstance(elem, ast.Expr) and hasattr(elem.value, 'args') and len(elem.value.args) > 0 \
            and isinstance(elem.value.args[0], ast.Call) \
            and hasattr(elem.value.args[0].func, 'id') and elem.value.args[0].func.id == 'Activation' \
            and hasattr(elem.value.args[0], 'args') \
            and isinstance(elem.value.args[0].args[0], ast.Str) \
            and elem.value.args[0].args[0].s in const.activation_functions:
        res = True
        type = 'L'

    return res, type, pos


def has_activation_func(elem):
    res = False
    def_type = 0
    pos = None

    # Samiy veroyarrrniy
    res, def_type, pos = is_activation_def_by_kwd(elem)

    if not res:
        res, def_type, pos = is_activation_def_by_layer(elem)
    # if not res:
    #     res, def_type, pos = is_activation_def_by_arg(elem)
    return res, def_type, pos


def is_activation_assignment(elem):
    """Check if the given element is an import of library

        Keyword arguments:
        elem -- part of ast node

        Returns: boolean
    """
    is_aa = False
    type = None
    pos = None

    if is_specific_call(elem, "add"):
        is_aa, type, pos = has_activation_func(elem)

    return is_aa, type, pos


def is_import(elem):
    """Check if the given element is an import of library

        Keyword arguments:
        elem -- part of ast node

        Returns: boolean
    """

    is_imp = False

    if isinstance(elem, ast.Import):
        is_imp = True

    return is_imp


def is_layer(elem):
    is_lyr = False

    return is_lyr


def is_optimiser_object(elem):
    """Check if the given element corresponds to a specific method call

        Keyword arguments:
        elem -- part of ast node
        call_type -- type of the call: fit, evaluate, add

        Returns: boolean
    """

    keras_optimiser = False
    definition_type = None

    if isinstance(elem, ast.Assign)\
            and isinstance(elem.value, ast.Call) \
            and ((hasattr(elem.value.func, "attr") and elem.value.func.attr in const.keras_optimisers)\
                or (hasattr(elem.value.func, "id") and elem.value.func.id in const.keras_optimisers))\
            :
        keras_optimiser = True
        definition_type = "object"

    return keras_optimiser, definition_type

def is_conv_layer_1(elem):
    type = None
    if isinstance(elem, ast.Assign) \
            and isinstance(elem.value, ast.Call) \
            and hasattr(elem.value.func, "attr") \
            and "Conv" in str(elem.value.func.attr):
        type = 1
    return type

def is_conv_layer_2(elem):
    type = None
    if isinstance(elem, ast.Expr) \
            and isinstance(elem.value, ast.Call) \
            and hasattr(elem.value, "args") and len(elem.value.args) > 0\
            and isinstance(elem.value.args[0], ast.Call) \
            and hasattr(elem.value.args[0].func, "attr") \
            and "Conv" in str(elem.value.args[0].func.attr):
        type = 2
    return type

def is_conv_layer_3(elem):
    type = None
    if isinstance(elem, ast.Assign) \
            and isinstance(elem.value, ast.Call) \
            and hasattr(elem.value.func, "id") \
            and "Conv" in str(elem.value.func.id):
        type = 3
    return type

def is_conv_layer_4(elem):
    type = None
    if isinstance(elem, ast.Expr) \
            and isinstance(elem.value, ast.Call) \
            and hasattr(elem.value, "args") and len(elem.value.args) > 0 \
            and isinstance(elem.value.args[0], ast.Call) \
            and hasattr(elem.value.args[0].func, "id") \
            and "Conv" in str(elem.value.args[0].func.id):
        type = 4
    return type

def is_conv_layer(elem):
    result = False

    t1 = is_conv_layer_1(elem)
    t2 = is_conv_layer_2(elem)
    t3 = is_conv_layer_3(elem)
    t4 = is_conv_layer_4(elem)

    if t1 or t2 or t3 or t4:
        result = True

    return result

def is_layer_1(elem):
    type = None
    if isinstance(elem, ast.Assign) \
            and isinstance(elem.value, ast.Call) \
            and hasattr(elem.value.func, "attr") \
            and str(elem.value.func.attr):
        type = 1
    return type

def is_layer_2(elem):
    type = None
    if isinstance(elem, ast.Expr) \
            and isinstance(elem.value, ast.Call) \
            and hasattr(elem.value, "args") and len(elem.value.args) > 0\
            and isinstance(elem.value.args[0], ast.Call) \
            and hasattr(elem.value.args[0].func, "attr") \
            and str(elem.value.args[0].func.attr):
        type = 2
    return type

def is_layer_3(elem):
    type = None
    if isinstance(elem, ast.Assign) \
            and isinstance(elem.value, ast.Call) \
            and hasattr(elem.value.func, "id") \
            and str(elem.value.func.id):
        type = 3
    return type

def is_layer_4(elem):
    type = None
    if isinstance(elem, ast.Expr) \
            and isinstance(elem.value, ast.Call) \
            and hasattr(elem.value, "args") and len(elem.value.args) > 0 \
            and isinstance(elem.value.args[0], ast.Call) \
            and hasattr(elem.value.args[0].func, "id") \
            and str(elem.value.args[0].func.id):
        type = 4
    return type

def is_layer(elem):
    result = False

    t1 = is_layer_1(elem)
    t2 = is_layer_2(elem)
    t3 = is_layer_3(elem)
    t4 = is_layer_4(elem)

    if t1 or t2 or t3 or t4:
        result = True

    return result







####################################################   ####################################################################
###################################### File MOD Functions ###############################################################


def rename_trained_model(file_path):
    """ Script that renames the file with trained model

        Keyword arguments:
        file_path -- path to the file
        ... params needed to constuct new name

        Returns: ...
    """
    if len(file_path) != 3:
        raise Exception("File path should be an array of 3 elements.")

    file_name = os.path.join(file_path[0], file_path[1])
    new_file_name = os.path.join(file_path[0], file_path[2])

    print(new_file_name)

    if (os.path.exists(file_name)):
        os.rename(file_name, new_file_name)
    # else:
    #     print("NO")
    #     print(os.path.exists(file_name))

    # except:
    #     print("Trained model rename failed")
    return True


def name_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """ Script that renames the file with trained model

        Keyword arguments:
        file_path -- path to the file
        ... params needed to constuct new name

        Returns: ...
    """
    return ''.join(random.choice(chars) for _ in range(size))


def save_scores(scores, model_name):
    """ Script that renames the file with trained model

        Keyword arguments:
        file_path -- path to the file
        ... params needed to constuct new name

        Returns: ...
    """
    file_name = model_name.replace(".h5", ".txt")
    f = open(file_name, "w")
    for x in scores:
        f.write(str(x) + "\n")
    f.close()

def save_scores2(scores, mutant_path):
    """ Script that renames the file with trained model

        Keyword arguments:
        file_path -- path to the file
        ... params needed to constuct new name

        Returns: ...
    """

    file_path = mutant_path[0] + "results/" + mutant_path[1].replace(".py", "") + ".txt"
    f = open(file_path, "a+")

    for ind, score in enumerate(scores):
        line = str(ind) +  " " + str(score) + "\n"
        f.write(line)
    f.close()

def save_scores_csv(scores, file_path, mutation_params = None):
    """ Script that renames the file with trained model

        Keyword arguments:
        file_path -- path to the file
        ... params needed to constuct new name

        Returns: ...
    """
    row_list = []

    for ind, score in enumerate(scores):
        row_list.append([ind+1, score[0], score[1]])

    with open(file_path, "w+", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)

def load_scores_from_csv(file_path):
    scores = []
    with open(file_path) as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            if any(x.strip() for x in row):
                scores.append((float(row[1]), float(row[2])))

    return scores

def concat_params_for_file_name(params):
    """ Script that renames the file with trained model

        Keyword arguments:
        file_path -- path to the file
        ... params needed to construct new name

        Returns: ...
    """
    list_params = ""
    for k, v in params.items():
        if any(abbrv in k for abbrv in const.mutation_params_abbrvs):
            list_params += str(v) + "_"

    if list_params:
        list_params = "_" + list_params[:-1]

    return list_params


def update_mutation_properties(mutation, param, new_value):
    """ Script that renames the file with trained model

        Keyword arguments:
        file_path -- path to the file
        ... params needed to constuct new name

        Returns: ...
    """
    params = getattr(props, mutation)

    keys = [key for key, value in params.items() if param in key.lower()]

    for key in keys:
        params[key] = new_value

def get_accuracy_list_from_scores(scores):
    scores_len = len(scores)
    accuracy_list = list(range(0, scores_len))
    for i in range (0, scores_len):
        accuracy_list[i] = scores[i][1]

    return accuracy_list


####################################################   ####################################################################
###################################### Mutation Functions ##############################################################

def model_from_config(model, tmp):
    if isinstance(model, KES):
        model = KS.from_config(tmp)
    elif isinstance(model, KEM):
        model = KM.from_config(tmp)
    elif isinstance(model, TKES):
        model = TKS.from_config(tmp)
    elif isinstance(model, TKEM):
        model = TKM.from_config(tmp)
    else:
        print("raise,log we have probllems")

    return model