import datetime
import traceback
from pathlib import Path
import torch
import numpy as np
from relnet.agent.base_agent import Agent
from relnet.evaluation.file_paths import FilePaths
from relnet.utils.config_utils import get_device_placement


class PyTorchAgent(Agent):
    # Determines batch size for experience replay
    DEFAULT_BATCH_SIZE = 50

    def __init__(self, environment):
        """
        :param environment: Graph environment agent in operating in
        """
        super().__init__(environment)
        self.enable_assertions = True
        # USed for logging history
        self.hist_out = None
        self.hist_out_raw = None
        self.hist_out_raw_test = None
        # Used for checking if a better significantly better validation loss has been found
        self.validation_change_threshold = 1e-7
        # Counter for checking if training with no improvement
        self.best_validation_changed_step = -1
        # Loss checker
        # self.best_validation_loss = float("inf")
        self.best_validation_loss = -float("inf")
        # Counter for position in buffer or dataset of graphs?
        self.pos = 0
        # Step count - not sure if its used
        self.step = 0

    def setup_graphs(self, train_g_list, validation_g_list):
        """
        Sets up many graphs for train/valid and objective function values
        :param train_g_list: List of training graphs
        :param validation_g_list: List of validation graphs
        """
        self.train_g_list = train_g_list
        self.validation_g_list = validation_g_list
        self.train_initial_obj_values = self.environment.get_objective_function_values(self.train_g_list)
        self.validation_initial_obj_values = self.environment.get_objective_function_values(self.validation_g_list)

    def setup_sample_idxes(self, dataset_size):
        """
        Initialises sampling from each graph uniformly?
        :param dataset_size: Size of the data set of graphs
        """
        self.sample_idxes = list(range(dataset_size))

    def advance_pos_and_sample_indices(self):
        """
        What are these indices used for??
        :return: Selected indices
        """
        # If reaches end of buffer/dataset, reset and shuffle
        if (self.pos + 1) * self.batch_size > len(self.sample_idxes):
            self.pos = 0
            np.random.shuffle(self.sample_idxes)
        # Samples a batch from dataset (is dataset a buffer or set of graphs) and advances position
        selected_idx = self.sample_idxes[self.pos * self.batch_size: (self.pos + 1) * self.batch_size]
        self.pos += 1
        return selected_idx

    def save_model_checkpoints(self):
        model_dir = self.checkpoints_path / self.model_identifier_prefix
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{self.algorithm_name}_agent.model"
        torch.save(self.net.state_dict(), model_path)

    def restore_model_from_checkpoint(self):
        model_path = self.checkpoints_path / self.model_identifier_prefix / f"{self.algorithm_name}_agent.model"
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint)

    def check_validation_loss(self,
                              step_number,
                              max_steps,
                              make_action_kwargs=None,
                              model_tag=None,
                              save_model_if_better=True):
        """
        Checks if current model has better validation loss and saves if required
        :param step_number: The training step number (why this and not self.step?)
        :param max_steps: Make check if on last step
        :param make_action_kwargs: key word arguments for getting validation
        :param model_tag: Tag for model name
        :param save_model_if_better: Bool to save better models if true
        """
        # Make check if on check step or final step
        if (step_number % self.validation_check_interval == 0) or (step_number == max_steps):
            # Get validation loss
            validation_loss = self.log_validation_loss(step_number, make_action_kwargs=make_action_kwargs)
            # If logging then log the relevant info
            if self.log_progress:
                self.logger.info(f"{model_tag if model_tag is not None else 'model'} validation performance:"
                                 f" {validation_loss: .4f} at step {step_number}.")
            # If loss if significantly better then log and save
            if (self.best_validation_loss < validation_loss) and \
                    ((validation_loss - self.best_validation_loss) > self.validation_change_threshold):
                # If logging then log the relevant info then save stats
                if self.log_progress:
                    self.logger.info(f"rejoice! found a better validation loss at step {step_number}.")
                self.best_validation_changed_step = step_number
                self.best_validation_loss = validation_loss
                # Save model
                if save_model_if_better:
                    if self.log_progress:
                        self.logger.info("saving model.")
                    self.save_model_checkpoints()

    def log_validation_loss(self, step, make_action_kwargs=None):
        """
        Calculates the logs the validation set loss
        :param step: Current step
        :param make_action_kwargs: args
        :return: Validation loss at current step
        """
        performance = self.eval(self.validation_g_list,
                                self.validation_initial_obj_values,
                                validation=True,
                                make_action_kwargs=make_action_kwargs)

        # If logging - do logs
        if self.log_tf_summaries:
            # Maybe should be in main imports but uses here to save memory if not required
            from tensorflow import Summary
            # Uses tensorflow to write a summary - do we need this?
            validation_summary = Summary(value=[
                Summary.Value(tag="validation_performance", simple_value=performance)])
            # Write to tensorbaord
            self.file_writer.add_summary(validation_summary, step)
            # Tries to write tensorboard info to disk
            try:
                self.file_writer.flush()
            # All exceptions - not super informative!
            except BaseException:
                # Makes note of error - unknown reason atm
                if self.logger is not None:
                    self.logger.warn("caught an exception when trying to flush TF data.")
                    self.logger.warn(traceback.format_exc())
        # If we have something stored in the history output
        if self.hist_out is not None:
            # Write our performance to it
            self.hist_out.write('%d,%.6f\n' % (step, performance))
            # Get the raw values and write to csv file (step, init, final)
            for init, final in zip(self.initial_obj_values, self.final_obj_values):
                self.hist_out_raw.write('%d,%.6f,%.6f\n' % (step, init, final))
            try:
                self.hist_out.flush()
                self.hist_out_raw.flush()
            except BaseException:
                if self.logger is not None:
                    self.logger.warn("caught an exception when trying to flush evaluation history.")
                    self.logger.warn(traceback.format_exc())
        # Return the validation loss
        return performance

    def log_test_performance(self,
                             test_performance,
                             baseline_performance,
                             initial_obj_values,
                             final_obj_values_test,
                             final_obj_values_baseline,
                             baseline_2_performance=None,
                             final_obj_values_baseline_2=None):

        if self.hist_out is not None:
            self.hist_out.write('Test set performance,%.6f\n' % (test_performance))
            self.hist_out.write('Random baseline test set performance,%.6f\n' % (baseline_performance))
            try:
                self.hist_out.flush()
            except BaseException:
                if self.logger is not None:
                    self.logger.warn("caught an exception when trying to flush evaluation history.")
                    self.logger.warn(traceback.format_exc())

        if self.hist_out_raw_test is not None:
            if baseline_2_performance is None:
                for init, test, base in zip(initial_obj_values, final_obj_values_test, final_obj_values_baseline):
                    self.hist_out_raw_test.write('%.6f,%.6f,%.6f\n' % (init, test, base))
            else:
                self.hist_out.write('Baseline test set performance,%.6f\n' % (baseline_2_performance))
                for init, test, base, base2 in zip(initial_obj_values,
                                                   final_obj_values_test,
                                                   final_obj_values_baseline,
                                                   final_obj_values_baseline_2):
                    self.hist_out_raw_test.write('%.6f,%.6f,%.6f, %.6f\n' % (init, test, base, base2))

    def update_test_performance(self,
                             test_performance,
                             baseline_performance,
                             initial_obj_values,
                             final_obj_values_test,
                             final_obj_values_baseline,
                             baseline_2_performance=None,
                             final_obj_values_baseline_2=None):

        for init, test, base, base2 in zip(initial_obj_values, final_obj_values_test,
                                           final_obj_values_baseline, final_obj_values_baseline_2):
            self.hist_out_raw_test.write('%.6f,%.6f,%.6f, %.6f\n' % (init, test, base, base2))
        try:
            self.hist_out_raw_test.flush()
        except BaseException:
            if self.logger is not None:
                self.logger.warn("caught an exception when trying to flush evaluation history.")
                self.logger.warn(traceback.format_exc())

    def log_oos_performance(self, init_obj, final_obj, n_nodes, graph_type):
        for init, final in zip(init_obj, final_obj):
            self.hist_out_raw_oos.write('%.6f,%.6f,%i,%s\n' % (init, final, n_nodes, graph_type))
        try:
            self.hist_out_raw_oos.flush()
        except BaseException:
            if self.logger is not None:
                self.logger.warn("caught an exception when trying to flush evaluation history.")
                self.logger.warn(traceback.format_exc())

    def print_model_parameters(self):
        """
        Print the current model parameters
        """
        param_list = self.net.parameters()
        for params in param_list:
            print(params.data)

    def check_stopping_condition(self, step_number, max_steps):
        # Check if exceeded max steps or too many steps with no validation improvement
        if (step_number >= max_steps) \
                or (step_number - self.best_validation_changed_step > self.max_validation_consecutive_steps):
            # If logging then update info
            if self.log_progress:
                self.logger.info("number steps exceeded or validation plateaued for too long, stopping training.")
                self.logger.info("restoring best model to use for predictions.")

            if self.restore_model_at_end is True:
                # Restore model
                self.restore_model_from_checkpoint()
            else:
                self.save_model_checkpoints()

            # If using tensorboard then close it
            if self.log_tf_summaries:
                self.file_writer.close()
            # Return true for end condition being met
            return True
        # Return false for end condition not being met
        return False

    def setup(self, options, hyperparams):
        """
        Sets up the parameters for the agents using inputs
        :param options: set up parameters for agent
        :param hyperparams: Doesn't seem to be uses here?
        """
        super().setup(options, hyperparams)
        # Sets batch size for sampling from experience replay
        if 'batch_size' in options:
            self.batch_size = options['batch_size']
        else:
            self.batch_size = self.DEFAULT_BATCH_SIZE
        # Sets validation check interval
        if 'validation_check_interval' in options:
            self.validation_check_interval = options['validation_check_interval']
        else:
            self.validation_check_interval = 100
        # Sets max steps with no improve
        if 'max_validation_consecutive_steps' in options:
            self.max_validation_consecutive_steps = options['max_validation_consecutive_steps']
        else:
            self.max_validation_consecutive_steps = 200000
        # Sets print option during training
        if 'pytorch_full_print' in options:
            if options['pytorch_full_print']:
                torch.set_printoptions(profile="full")
        # Not sure what this is used for
        if 'enable_assertions' in options:
            self.enable_assertions = options['enable_assertions']
        # Information of which model we are using
        if 'model_identifier_prefix' in options:
            self.model_identifier_prefix = options['model_identifier_prefix']
        else:
            self.model_identifier_prefix = FilePaths.DEFAULT_MODEL_PREFIX
        # Keep final model
        if 'restore_model' in options:
            self.restore_model = options['restore_model']
        else:
            self.restore_model = False
        # Sets model save path
        if 'models_path' in options:
            self.models_path = Path(options['models_path'])
        else:
            self.models_path = Path.cwd() / FilePaths.MODELS_DIR_NAME
        # Defines path for saving models
        self.checkpoints_path = self.models_path / FilePaths.CHECKPOINTS_DIR_NAME
        # If using tensorboard set up
        if 'log_tf_summaries' in options and options['log_tf_summaries']:
            self.summaries_path = self.models_path / FilePaths.SUMMARIES_DIR_NAME
            # Import relevant tensorflow modules
            from tensorflow import Graph
            from tensorflow.summary import FileWriter
            self.log_tf_summaries = True
            summary_run_dir = self.get_summaries_run_path()
            # Set the file writer
            self.file_writer = FileWriter(summary_run_dir, Graph())
        else:
            self.log_tf_summaries = False
        if 'discount' in options:
            self.discount = options['discount']
        else:
            self.discount = 0.99
        if 'soft' in options:
            self.soft = options['soft']
            if 'tau' in options:
                self.tau = options['tau']
            else:
                self.tau = 0.01
        else:
            self.soft = False
        if 'restore_end' in options:
            self.restore_model_at_end = options['restore_end']
        else:
            self.restore_model_at_end = True

    def get_summaries_run_path(self):
        """
        Uses current date and time to save summaries
        :return: The path to save summaries to
        """
        now = datetime.datetime.now().strftime(FilePaths.DATE_FORMAT)
        return self.summaries_path / f"{self.model_identifier_prefix}-run-{now}"

    def setup_histories_file(self):
        """
        Sets up the file and path for saving history
        """
        # Sets the path
        self.eval_histories_path = self.models_path / FilePaths.EVAL_HISTORIES_DIR_NAME

        model_history = self.eval_histories_path / self.model_identifier_prefix
        model_history.mkdir(parents=True, exist_ok=True)

        model_history_filename = model_history / FilePaths.construct_history_file_name('validation')
        # Uses Pathlib to make the path to the file
        model_history_file = Path(model_history_filename)
        # Organises the file and unlinks it from path if required
        if model_history_file.exists():
            model_history_file.unlink()
        # Open the file for output
        self.hist_out = open(model_history_filename, 'a')
        self.hist_out.write('Step, SW Improvement\n')

        # Set up test and baselines
        raw_history_filename = model_history / FilePaths.construct_history_file_name('raw')
        raw_history_filename = Path(raw_history_filename)
        if raw_history_filename.exists():
            raw_history_filename.unlink()
        self.hist_out_raw = open(raw_history_filename, 'a')
        self.hist_out_raw.write('Step, Start SW, Final SW\n')

        # Set up test and baselines
        raw_test_history_filename = model_history / FilePaths.construct_history_file_name('test')
        raw_test_history_filename = Path(raw_test_history_filename)
        if raw_test_history_filename.exists():
            raw_test_history_filename.unlink()
        self.hist_out_raw_test = open(raw_test_history_filename, 'a')
        self.hist_out_raw_test.write('Start SW, Test SW, Random Baseline SW, Policy Baseline\n')

    def update_test_history(self):
        self.eval_histories_path = self.models_path / FilePaths.EVAL_HISTORIES_DIR_NAME
        model_history = self.eval_histories_path / self.model_identifier_prefix
        model_history.mkdir(parents=True, exist_ok=True)
        raw_test_history_filename = model_history / FilePaths.construct_history_file_name('test')
        raw_test_history_filename = Path(raw_test_history_filename)
        if raw_test_history_filename.exists():
            raw_test_history_filename.unlink()
        self.hist_out_raw_test = open(raw_test_history_filename, 'a')
        self.hist_out_raw_test.write('Start SW, Test SW, Random Baseline SW, Policy Baseline\n')

    def setup_out_of_sample(self):
        self.eval_histories_path = self.models_path / FilePaths.EVAL_HISTORIES_DIR_NAME
        model_history = self.eval_histories_path / self.model_identifier_prefix
        model_history.mkdir(parents=True, exist_ok=True)
        raw_oos_history_filename = model_history / FilePaths.construct_history_file_name('out_of_sample')
        raw_oos_history_filename = Path(raw_oos_history_filename)
        if raw_oos_history_filename.exists():
            raw_oos_history_filename.unlink()
        self.hist_out_raw_oos = open(raw_oos_history_filename, 'a')
        self.hist_out_raw_oos.write('Start SW, Final SW, Nodes, Graph Type\n')

    def finalize(self):
        # Closes out the hist out file for writing
        if self.hist_out is not None and not self.hist_out.closed:
            self.hist_out.close()
        if self.hist_out_raw is not None and not self.hist_out_raw.closed:
            self.hist_out_raw.close()
