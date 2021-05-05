import utils.execute_mutants as exc
import utils.properties as props

from utils.mutation_utils import *
from utils.logger_setup import setup_logger
from run_deepcrime_properties import read_properties


def mutate_model():
    logger = setup_logger(__name__)

    dc_props = read_properties()
    file_path = dc_props['subject_path']
    model_name = dc_props['subject_name']
    # list of mutations operators to be applied. Full list can be found in utils.constants
    mutations = dc_props['mutations']
    props.model_name = model_name

    print("Model Name "+ model_name)
    print("Path " + file_path)
    # raise Exception('exception')

    mutation_types = ['D', 'H']

    save_path_prepared = os.path.join(const.save_paths["prepared"], model_name + "_saved.py")
    save_path_trained = os.path.join("..", const.save_paths["trained"], model_name + "_trained.h5")

    prepared_path = os.path.join(const.save_paths["prepared"])

    if not os.path.exists(prepared_path):
        try:
            os.makedirs(prepared_path)
        except OSError as e:
            logger.error("Unable to create folder for mutated models:" + str(e))


    trained_path = os.path.join(const.save_paths["trained"])

    if not os.path.exists(trained_path):
        try:
            os.makedirs(trained_path)
        except OSError as e:
            logger.error("Unable to create folder for mutated models:" + str(e))


    prepare_model(file_path, save_path_prepared, save_path_trained, mutation_types)


    mutants_path = os.path.join(const.save_paths["mutated"], model_name)
    results_path = os.path.join(mutants_path, "results")
    stats_path = os.path.join(results_path, "stats")


    if not os.path.exists(mutants_path):
        try:
            os.makedirs(mutants_path)
        except OSError as e:
            logger.error("Unable to create folder for mutated models:" + str(e))


    if not os.path.exists(results_path):
        try:
            os.makedirs(results_path)
        except OSError as e:
            logger.error("Unable to create folder for mutated models:" + str(e))


    if not os.path.exists(stats_path):
        try:
            os.makedirs(stats_path)
        except OSError as e:
            logger.error("Unable to create folder for mutated models:" + str(e))

    for mutation in mutations:
        logger.info("Starting mutation %s", mutation)
        save_path_mutated = os.path.join(mutants_path, model_name + "_" + mutation + "_mutated")

        try:
            mutationClass = create_mutation(mutation)

            mutationClass.mutate(save_path_prepared, save_path_mutated)
        except LookupError as e:
            logger.info("Unable to apply the mutation for mutation %s. See technical logs for details. ", mutation)
            logger.error("Was not able to create a class for mutation %s: " + str(e), mutation)
        except Exception as e:
            logger.info("Unable to apply the mutation for mutation %s. See technical logs for details. ", mutation)
            logger.error("Unable to apply the mutation for mutation %s: " + str(e), mutation)


        logger.info("Finished mutation %s", mutation)


    exc.execute_original_model(file_path, results_path)

    exc.execute_mutants(mutants_path, mutations)


