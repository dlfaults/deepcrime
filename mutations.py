import ast
from copy import deepcopy

from utils import mutation_utils as mu
from utils.logger_setup import setup_logger
import utils.properties as props


logger = setup_logger(__name__)

class Mutation():
    applyOnce = True
    mutationName = None

    def get_model_params_td(self, elem):
        """Extract a dict of params such as x_train, y_train
           needed for mutation from a given node

            Keyword arguments:
            elem -- part of ast node

            Returns: dict (params)
        """

        params = {}

        # check fo the params in element arguments or keywords
        if hasattr(elem.value, 'args') and len(elem.value.args) > 0:
            # params["x_train"] = elem.value.args[0].id
            # params["y_train"] = elem.value.args[1].id
            params["x_train"] = elem.value.args[0]
            params["y_train"] = elem.value.args[1]
        elif hasattr(elem.value, 'keywords') and len(elem.value.keywords) > 0:
            for x in elem.value.keywords:
                # print(x)
                if x.arg == 'x':
                    # params["x_train"] = x.value.id
                    params["x_train"] = x.value
                if x.arg == 'y':
                    # params["y_train"] = x.value.id
                    params["y_train"] = x.value
        else:
            logger.error("Mutation.get_model_params_td AST node does not have arguments or keywords")

        return params

    def get_model_params_hp(self, elem):
        """Extract a dict of params such as number of epochs, batch size, etc.
           needed for mutation from a given node

            Keyword arguments:
            elem -- part of ast node

            Returns: dict (params)
        """

        params = {}

        if hasattr(elem.value, 'keywords') and len(elem.value.keywords) > 0:
            for k in elem.value.keywords:
                if type(k.value) == ast.Name:
                    params[k.arg] = k.value.id
                elif type(k.value) == ast.Num:
                    params[k.arg] = k.value.n
                elif type(k.value) == ast.Str:
                    params[k.arg] = k.value.s
                elif type(k.value) == ast.Constant:
                    params[k.arg] = k.value.value
                elif type(k.value) == ast.Attribute:
                    params[k.arg] = 'attr'
        else:
            logger.error("Mutation.get_model_params_hp AST node does not have keywords")

        return params

    def add_keyword(self, elem, kwd_name, kwd_value):
        try:
            elem.value.keywords.append(ast.keyword(arg=kwd_name, value=ast.Name(id=kwd_value, ctx=ast.Load())))
        except Exception:
            logger.error("Mutation.add_keyword adding keyword to AST node failed")

#TODO: Check if we need this method
    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}
        # params["mutation_name"] = mutation_name

        # TODO: write the param extraction
        # FOR NOW it will be like this, after, we read from the file given the mutation name

        params["module_name"] = "dummy_operators"
        params["operator_name"] = "dummy_operator"

        return params

    def mutate(self, file_path, save_path_mutated):
        """ Apply mutation to the given model.

        Keyword arguments:
        file_path -- path to the py file with model
        save_path_mutated -- path for the mutated model to be saved to

        Returns: -
        """

        # Parse the code for ast tree
        try:
            with open(file_path, "r") as source:
                tree = ast.parse(source.read())
        except Exception as e:
            logger.error("Mutation.mutate parse AST tree failed" + str(e))
            raise

        try:
            was_annotated = self.mutate_annotated(tree, save_path_mutated)
        except Exception as e:
            logger.error("Mutation.mutate_annotated failed:" + str(e))

        try:
            if not was_annotated:
                self.mutate_automatically(tree, save_path_mutated)
        except Exception as e:
            logger.error("Mutation.mutate_automatically failed" + str(e))

    def mutate_annotated(self, tree, save_path_mutated):
        """ Apply mutation to the given model with annotated params.

        Keyword arguments:
        tree -- AST tree
        save_path_mutated -- path for the mutated model to be saved to

        Returns: -
        """
        application_cnt = 0
        model_params_ann = {}
        was_annotated = False

        mutation_params = getattr(props, self.mutationName)

        for x in mutation_params["annotation_params"]:
            model_params_ann[x] = None

        # Look for the right place to insert the mutation
        # Commented is the functionality to apply the same mutation a number of times (Not needed atm)
        for node in ast.walk(tree):
            if hasattr(node, 'body') and isinstance(node.body, list):
                for ind, x in enumerate(node.body):
                    # check for annotation
                    mu.check_for_annotation(x, model_params_ann)
                    # if all annotations find then insert mutation
                    if not None in model_params_ann.values() \
                            and len(model_params_ann) > 0:
                    #     original_x = deepcopy(node.body[ind])
                    #
                        self.apply_mutation(node, x, ind+1, model_params_ann)

                        ast.fix_missing_locations(tree)
                        save_path_mutated_cnt = save_path_mutated + str(application_cnt) + '.py'
                        mu.unparse_tree(tree, save_path_mutated_cnt)

                        was_annotated = True

                        break
                    #     if self.applyOnce:
                    #         break
                    #     else:
                    #         application_cnt += 1
                    #         node.body[ind] = original_x
            if not None in model_params_ann.values():
                break

        return was_annotated

    def mutate_automatically(self, tree, save_path_mutated):
        """ Apply mutation to the given model (no annotations - automated)

        Keyword arguments:
        tree -- AST tree
        save_path_mutated -- path for the mutated model to be saved to

        Returns: -
        """

        application_cnt = 0
        # Look for the right place to insert the mutation
        for node in ast.walk(tree):
            if hasattr(node, 'body') and isinstance(node.body, list):
                for ind, x in enumerate(node.body):
                    if self.is_target_node(x):
                        original_x = deepcopy(node.body[ind])
                        self.apply_mutation(node, x, ind)

                        ast.fix_missing_locations(tree)
                        save_path_mutated_cnt = save_path_mutated + str(application_cnt) + '.py'
                        mu.unparse_tree(tree, save_path_mutated_cnt)

                        if self.applyOnce:
                            break
                        else:
                            application_cnt += 1
                            node.body[ind] = original_x

    def apply_mutation(self, node, elem, ind, model_params = None):
        # Each class has its own implementation
        return None

    def is_target_node(self, elem):
        # Each class has its own implementation
        return None


#########################################
###########   Training DATA  ############

class ChangeLabelTDMut(Mutation):
    mutationName = "change_label"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def get_model_params(self, elem):
        return self.get_model_params_td(elem)

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}
        # params["mutation_name"] = mutation_name

        # TODO: write the param extraction
        # FOR NOW it will be like this, after, we read from the file given the mutation name

        label = None
        percentage = -1

        params["module_name"] = "training_data_operators"
        params["operator_name"] = "operator_change_labels"
        params["label"] = "properties.change_label['change_label_label']"
        params["percentage"] = "properties.change_label['change_label_pct']"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        # Get model specific params
        if model_params_ann:
            model_params = model_params_ann
            model_params["y_train"] = ast.Name(id=model_params["y_train"], ctx=ast.Store())
        else:
            model_params = self.get_model_params(elem)
        # Get mutation specific params
        mutation_params = self.get_mutation_params()
        # print(mutation_params)

        mutation_node = ast.Assign(targets=[
                                            # ast.Name(id=model_params["y_train"], ctx=ast.Store()),
                                            model_params["y_train"],
                                            ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[
                                             # ast.Name(id=model_params["y_train"], ctx=ast.Load()),
                                             # ast.Str(s=mutation_params["label"]),
                                             # ast.Num(n=mutation_params["percentage"]), ],
                                             model_params["y_train"],
                                             ast.Name(id=mutation_params["label"], ctx=ast.Load()),
                                             ast.Name(id=mutation_params["percentage"], ctx=ast.Load()),],
                                       keywords=[]))

        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):
        # generate a mutation call
        mutation_node = self.generate_mutation_node(elem, model_params_ann)
        # insert a mutation call
        node.body.insert(ind, mutation_node)
        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params_ann = None):
        self.insert_mutation(node, elem, ind, model_params_ann)


class DeleteTDMut(Mutation):
    mutationName = "delete_training_data"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def get_model_params(self, elem):
        return self.get_model_params_td(elem)

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}
        # params["mutation_name"] = mutation_name

        # TODO: write the param extraction
        # FOR NOW it will be like this, after, we read from the file given the mutation name

        params["module_name"] = "training_data_operators"
        params["operator_name"] = "operator_delete_training_data"
        params["percentage"] = "properties.delete_training_data['delete_train_data_pct']"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        # Get model specific params
        if model_params_ann:
            model_params = model_params_ann
            model_params["x_train"] = ast.Name(id=model_params["x_train"], ctx=ast.Store())
            model_params["y_train"] = ast.Name(id=model_params["y_train"], ctx=ast.Store())
        else:
            model_params = self.get_model_params(elem)

        # Get mutation specific params
        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Tuple(elts=[
        #         ast.Name(id=model_params["x_train"], ctx=ast.Store()),
        #         ast.Name(id=model_params["y_train"], ctx=ast.Store()),
                model_params["x_train"],
                model_params["y_train"],
            ], ctx=ast.Store()),
            ],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                       attr=mutation_params["operator_name"],
                                       ctx=ast.Load()),
                    args=[
                          # ast.Name(id=model_params["x_train"], ctx=ast.Load()),
                          # ast.Name(id=model_params["y_train"], ctx=ast.Load()),
                          model_params["x_train"],
                          model_params["y_train"],
                          ast.Name(id=mutation_params["percentage"], ctx=ast.Load()), ],
                    keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):
        # generate a mutation call
        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        # insert a mutation call
        node.body.insert(ind, mutation_node)
        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params_ann = None):
        self.insert_mutation(node, elem, ind, model_params_ann)

class OutputClassesOverlapTDMUT(Mutation):
    mutationName = "make_output_classes_overlap"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def get_model_params(self, elem):
        return self.get_model_params_td(elem)

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}
        # params["mutation_name"] = mutation_name

        # TODO: write the param extraction
        # FOR NOW it will be like this, after, we read from the file given the mutation name

        params["module_name"] = "training_data_operators"
        params["operator_name"] = "operator_make_output_classes_overlap"
        params["percentage"] = "properties.make_output_classes_overlap['make_output_classes_overlap_pct']"

        return params

    def generate_mutation_node(self, elem, model_params_ann=None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        # Get model specific params
        if model_params_ann:
            model_params = model_params_ann
            model_params["x_train"] = ast.Name(id=model_params["x_train"], ctx=ast.Store())
            model_params["y_train"] = ast.Name(id=model_params["y_train"], ctx=ast.Store())
        else:
            model_params = self.get_model_params(elem)

        # Get mutation specific params
        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Tuple(elts=[
            # ast.Name(id=model_params["x_train"], ctx=ast.Store()),
            # ast.Name(id=model_params["y_train"], ctx=ast.Store()),
            model_params["x_train"],
            model_params["y_train"],
        ], ctx=ast.Store()),
        ],
            value=ast.Call(
                func=ast.Attribute(value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                   attr=mutation_params["operator_name"],
                                   ctx=ast.Load()),
                args=[
                      # ast.Name(id=model_params["x_train"], ctx=ast.Load()),
                      # ast.Name(id=model_params["y_train"], ctx=ast.Load()),
                      model_params["x_train"],
                      model_params["y_train"],
                      ast.Name(id=mutation_params["percentage"], ctx=ast.Load()), ],
                keywords=[]))

        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):
        # generate a mutation call
        mutation_node = self.generate_mutation_node(elem, model_params_ann)
        # insert a mutation call
        node.body.insert(ind, mutation_node)
        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params_ann = None):
        self.insert_mutation(node, elem, ind, model_params_ann)

class UnbalanceTDMut(Mutation):
    mutationName = "unbalance_train_data"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def get_model_params(self, elem):
        return self.get_model_params_td(elem)

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}
        # params["mutation_name"] = mutation_name

        # TODO: write the param extraction
        # FOR NOW it will be like this, after, we read from the file given the mutation name

        params["module_name"] = "training_data_operators"
        params["operator_name"] = "unbalance_training_data"
        params["percentage"] = "properties.unbalance_train_data['unbalance_train_data_pct']"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        # Get model specific params
        if model_params_ann:
            model_params = model_params_ann
            model_params["x_train"] = ast.Name(id=model_params["x_train"], ctx=ast.Store())
            model_params["y_train"] = ast.Name(id=model_params["y_train"], ctx=ast.Store())
        else:
            model_params = self.get_model_params(elem)

        # Get mutation specific params
        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Tuple(elts=[
                # ast.Name(id=model_params["x_train"], ctx=ast.Store()),
                # ast.Name(id=model_params["y_train"], ctx=ast.Store()),
                model_params["x_train"],
                model_params["y_train"],
            ], ctx=ast.Store()),
            ],
                value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                       attr=mutation_params["operator_name"],
                                       ctx=ast.Load()),
                    args=[
                          # ast.Name(id=model_params["x_train"], ctx=ast.Load()),
                          # ast.Name(id=model_params["y_train"], ctx=ast.Load()),
                          model_params["x_train"],
                          model_params["y_train"],
                          ast.Name(id=mutation_params["percentage"], ctx=ast.Load()), ],
                    keywords=[]))

        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):
        # generate a mutation call
        mutation_node = self.generate_mutation_node(elem, model_params_ann)
        # insert a mutation call
        node.body.insert(ind, mutation_node)
        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params_ann = None):
        self.insert_mutation(node, elem, ind, model_params_ann)


class AddNoiseTDMut(Mutation):
    mutationName = "add_noise"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def get_model_params(self, elem):
        return self.get_model_params_td(elem)

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}
        # params["mutation_name"] = mutation_name

        # TODO: write the param extraction
        # FOR NOW it will be like this, after, we read from the file given the mutation name

        params["module_name"] = "training_data_operators"
        params["operator_name"] = "operator_add_noise_to_training_data"
        params["percentage"] = "properties.add_noise['add_noise_pct']"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        # Get model specific params
        if model_params_ann:
            model_params = model_params_ann
            model_params["x_train"] = ast.Name(id=model_params["x_train"], ctx=ast.Store())
        else:
            model_params = self.get_model_params(elem)

        # Get mutation specific params
        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[
                                        # ast.Name(id=model_params["x_train"], ctx=ast.Store()),
                                            model_params["x_train"],
                                            ],
                                       value=ast.Call(
                                           func=ast.Attribute(
                                               value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                               attr=mutation_params["operator_name"],
                                               ctx=ast.Load()),
                                           args=[
                                                 # ast.Name(id=model_params["x_train"], ctx=ast.Load()),
                                                 model_params["x_train"],
                                                 ast.Name(id=mutation_params["percentage"], ctx=ast.Load()), ],
                                           keywords=[]))

        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):
        # generate a mutation call
        mutation_node = self.generate_mutation_node(elem, model_params_ann)
        # insert a mutation call
        node.body.insert(ind, mutation_node)
        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params_ann = None):
        self.insert_mutation(node, elem, ind, model_params_ann)


#########################################
############   HYPERPARAMS  #############

class ChangeLearnRateHPMut(Mutation):
    mutationName = "change_learning_rate"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, "compile")

    def get_model_params(self, elem):
        params = {}
        return params

    def get_mutation_params(self, optimiser_name = None):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "hyperparams_operators"
        params["operator_name"] = "operator_change_learning_rate"

        return params

    def perform_mutation(self, elem):
        params = self.get_mutation_params()

        for keyword in elem.value.keywords:
            if keyword.arg == "optimizer":
                keyword.value = ast.Call(func=ast.Attribute(value=ast.Name(id=params["module_name"], ctx=ast.Load()),
                                    attr=params["operator_name"], ctx=ast.Load()),
                                    args=[keyword.value,],
                                    keywords=[])

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.perform_mutation(elem)

class ChangeBatchSizeHPMut(Mutation):
    mutationName = "change_batch_size"

    def dummy(self):#__init__
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def get_model_params(self, elem):
        return self.get_model_params_hp(elem)

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "hyperparams_operators"
        params["operator_name"] = "operator_change_batch_size"

        return params

    def generate_mutation_node(self, elem, model_params):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        mutation_node = ast.Assign(targets=[
                ast.Name(id=model_params["batch_size"], ctx=ast.Store()), ],
                value=ast.Subscript(
                value=ast.Attribute(value=ast.Name(id='properties', ctx=ast.Load()), attr='change_batch_size',
                                    ctx=ast.Load()),
                slice=ast.Index(value=ast.Str(s='batch_size')), ctx=ast.Load()))

        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params):
        # generate a mutation call
        mutation_node = self.generate_mutation_node(elem, model_params)
        # insert a mutation call
        node.body.insert(ind, mutation_node)
        is_inserted = True
        return None

    def perform_mutation(self, elem):
        for keyword in elem.value.keywords:
            if keyword.arg == "batch_size":
                keyword.value = ast.Name(id="properties.change_batch_size['batch_size']", ctx=ast.Load())

    def apply_mutation(self, node, elem, ind, model_params = None):
        # Get model.fit specific params
        model_params = self.get_model_params(elem)

        batch_size = model_params.get("batch_size")

        if not props.change_batch_size["applicable"]:
            print("Change batch size in not applicable")
        elif batch_size is None:
            self.add_keyword(elem, "batch_size", "properties.change_batch_size['batch_size']")
        elif isinstance(batch_size, str):
            self.insert_mutation(node, elem, ind, model_params)
        elif isinstance(batch_size, int):
            self.perform_mutation(elem)
        else:
            print("Unknown batch size value")


class ChangeEpochsHPMut(Mutation):
    mutationName = "change_epochs"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def get_model_params(self, elem):
        return self.get_model_params_hp(elem)

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "hyperparams_operators"
        params["operator_name"] = "operator_change_epochs"

        return params


    def perform_mutation(self, elem):
        for keyword in elem.value.keywords:
            if keyword.arg == "epochs":
                keyword.value = ast.Name(id="properties.change_epochs['pct']", ctx=ast.Load())

    def apply_mutation(self, node, elem, ind, model_params = None):
        # Get model.fit specific params
        model_params = self.get_model_params(elem)

        epochs = model_params.get("epochs")

        if epochs is None:
            self.add_keyword(elem, "epochs", "properties.change_epochs['pct']")
        else:
            self.perform_mutation(elem)


class DisableBatchingHPMut(Mutation):
    mutationName = "disable_batching"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def get_model_params(self, elem):
        return self.get_model_params_hp(elem)

    def perform_mutation(self, elem):
        model_params = self.get_model_params(elem)

        batch_size = model_params.get("batch_size")

        if batch_size is None:
            self.add_keyword(elem, "batch_size", "properties.model_properties['x_train_len']")
        elif isinstance(batch_size, (str, int)):
            for keyword in elem.value.keywords:
                if keyword.arg == "batch_size":
                    keyword.value = ast.Name(id="properties.model_properties['x_train_len']", ctx=ast.Load())
        else:
            print("Unknown batch size value")

    def apply_mutation(self, node, elem, ind):
        if not props.disable_batching["applicable"]:
            print("Disable data batching size in not applicable")
        else:
            self.perform_mutation(elem)


#########################################
###########   Activation Function  ############

class ChangeActivationAFMut(Mutation):
    mutationName = "change_activation_function"

    # applyOnce = False

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "activation_function_operators"
        params["operator_name"] = "operator_change_activation_function"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)

class RemoveActivationAFMut(Mutation):
    mutationName = "remove_activation_function"

    # applyOnce = False

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "activation_function_operators"
        params["operator_name"] = "operator_remove_activation_function"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)

class AddActivationAFMut(Mutation):
    mutationName = "add_activation_function"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "activation_function_operators"
        params["operator_name"] = "operator_add_activation_function"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)


#########################################
###########   Optimiser  #################

class ChangeOptimisationFunction(Mutation):
    mutationName = "change_optimisation_function"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, "compile")

    def get_model_params(self, elem):
        params = {}
        return params

    def get_mutation_params(self, optimiser_name = None):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "optimiser_operators"
        params["operator_name"] = "operator_change_optimisation_function"

        return params

    def perform_mutation(self, elem):
        params = self.get_mutation_params()

        for keyword in elem.value.keywords:
            if keyword.arg == "optimizer":
                keyword.value = ast.Call(func=ast.Attribute(value=ast.Name(id=params["module_name"], ctx=ast.Load()),
                                    attr=params["operator_name"], ctx=ast.Load()),
                                    args=[keyword.value,],
                                    keywords=[])

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.perform_mutation(elem)


class ChangeGradientClip(Mutation):
    mutationName = "change_gradient_clip"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        result, type = mu.is_optimiser_object(elem)
        print(result)
        return result

    def get_model_params(self, elem):
        params = {}
        return params

    def get_mutation_params(self, optimiser_name = None):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}
        return params

    def perform_mutation(self, elem):
        if hasattr(elem.value, 'keywords') and len(elem.value.keywords) > 0:
            for k in elem.value.keywords:
                if k.arg == 'clipnorm':
                    k.value = ast.Name(id="properties.change_gradient_clip['clipnorm']", ctx=ast.Load())
                if k.arg == 'clipvalue':
                    k.value = ast.Name(id="properties.change_gradient_clip['clipvalue']", ctx=ast.Load())
        else:
            # TODO: add errrror
            print("we have a problem here")

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.perform_mutation(elem)


#########################################
###########   Validation  #################

class RemoveValidationSet(Mutation):
    mutationName = "remove_validation_set"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def perform_mutation(self, elem):
        if hasattr(elem.value, 'keywords') and len(elem.value.keywords) > 0:
            for k in elem.value.keywords:
                if k.arg == 'validation_data':
                    k.value = ast.NameConstant(value=None)
                if k.arg == 'validation_split':
                    k.value = ast.Num(n=0.0)
        else:
            print("we have a problem here")

        return None

    def apply_mutation(self, node, elem, ind):
        self.perform_mutation(elem)


#########################################
###########   EarlyStopping  #################

class ChangeEarlyStoppingPatience(Mutation):
    mutationName = "change_earlystopping_patience"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_training_call(elem)

    def get_model_params(self, elem):
        params = {}
        callbacks = None

        if hasattr(elem.value, 'keywords') and len(elem.value.keywords) > 0:
            for k in elem.value.keywords:
                if k.arg == 'callbacks':
                    callbacks = k.value

        params["callbacks"] = callbacks
        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}
        # params["mutation_name"] = mutation_name

        # TODO: write the param extraction
        # FOR NOW it will be like this, after, we read from the file given the mutation name

        params["module_name"] = "training_process_operators"
        params["operator_name"] = "operator_change_patience"

        return params

    def perform_mutation(self, elem):
        params = self.get_mutation_params()
        for keyword in elem.value.keywords:
            if keyword.arg == "callbacks":
                keyword.value = ast.Call(func=ast.Attribute(value=ast.Name(id=params["module_name"], ctx=ast.Load()),
                                         attr=params["operator_name"], ctx=ast.Load()),
                                         args=[keyword.value, ],
                                         keywords=[])

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.perform_mutation(elem)



#########################################
###########   Bias  #################

class AddBiasMut(Mutation):
    mutationName = "add_bias"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "bias_operators"
        params["operator_name"] = "operator_add_bias"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)


class RemoveBiasMut(Mutation):
    mutationName = "remove_bias"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "bias_operators"
        params["operator_name"] = "operator_remove_bias"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)


#########################################
###########   Loss  #################

class ChangeLossFunction(Mutation):
    mutationName = "change_loss_function"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, "compile")

    def get_model_params(self, elem):
        params = {}
        return params

    def get_mutation_params(self, optimiser_name = None):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "loss_operators"
        params["operator_name"] = "operator_change_loss_function"

        return params

    def perform_mutation(self, elem):
        params = self.get_mutation_params()

        for keyword in elem.value.keywords:
            if keyword.arg == "loss":
                keyword.value = ast.Call(func=ast.Attribute(value=ast.Name(id=params["module_name"], ctx=ast.Load()),
                                    attr=params["operator_name"], ctx=ast.Load()),
                                    args=[keyword.value,],
                                    keywords=[])

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.perform_mutation(elem)

#########################################
###########   Dropout  #################

class ChangeDropoutRate(Mutation):
    mutationName = "change_dropout_rate"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "dropout_operators"
        params["operator_name"] = "operator_change_dropout_rate"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)


#########################################
###########   Weights  #################

class ChangeWeightsInitialisation(Mutation):
    mutationName = "change_weights_initialisation"

    # applyOnce = False

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "weights_operators"
        params["operator_name"] = "operator_change_weights_initialisation"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)


class ChangeWeightsRegularisation(Mutation):
    mutationName = "change_weights_regularisation"

    # applyOnce = False

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "weights_operators"
        params["operator_name"] = "operator_change_weights_regularisation"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)


class RemoveWeightsRegularisation(Mutation):
    mutationName = "remove_weights_regularisation"

    # applyOnce = False

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "weights_operators"
        params["operator_name"] = "operator_remove_weights_regularisation"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)


class AddWeightsRegularisation(Mutation):
    mutationName = "add_weights_regularisation"

    def dummy(self):
        print("Mutation Initialised")

    def is_target_node(self, elem):
        return mu.is_specific_call(elem, 'compile')

    def get_model_params(self, elem):
        params = {}

        if isinstance(elem.value.func, ast.Attribute) \
            and hasattr(elem.value.func.value, 'id'):
            params["model_name"] = elem.value.func.value.id
        else:
            print("log, we have a problem")

        return params

    def get_mutation_params(self):
        """Extract a dict of params needed for mutation from a params file

            Keyword arguments:
            mutation_name -- name of the mutation

            Returns: dics (params)
        """

        params = {}

        params["module_name"] = "weights_operators"
        params["operator_name"] = "operator_add_weights_regularisation"

        return params

    def generate_mutation_node(self, elem, model_params_ann = None):
        """Generate a mutation node

            Keyword arguments:
            mutation_name -- name of a mutation (str)
            model_params -- params needed to build a mutation node. depend on the model (list)

            Returns: ast node (mutation_node)
        """

        model_params = self.get_model_params(elem)

        mutation_params = self.get_mutation_params()

        mutation_node = ast.Assign(targets=[ast.Name(id=model_params["model_name"], ctx=ast.Store()), ],
                                   value=ast.Call(
                                       func=ast.Attribute(
                                           value=ast.Name(id=mutation_params["module_name"], ctx=ast.Load()),
                                           attr=mutation_params["operator_name"],
                                           ctx=ast.Load()),
                                       args=[ast.Name(id=model_params["model_name"], ctx=ast.Load()), ],
                                       keywords=[]))
        return mutation_node

    def insert_mutation(self, node, elem, ind, model_params_ann = None):

        mutation_node = self.generate_mutation_node(elem, model_params_ann)

        node.body.insert(ind, mutation_node)

        is_inserted = True
        return None

    def apply_mutation(self, node, elem, ind, model_params = None):
        self.insert_mutation(node, elem, ind)