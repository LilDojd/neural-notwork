# Copyright (c) 2017 Jes Frellsen and Wouter Boomsma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from sklearn.metrics import classification_report as report


class BCOLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == '__main__':
    import glob
    import os
    import numpy as np
    from Models import models
    from batch_factory import BatchFactory
    from sklearn.utils import shuffle

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--high-res-input-dir", dest="high_res_features_input_dir", required=True,
                        help="Location of input files containing high-res features")
    parser.add_argument("--test-set-fraction",
                        help="Fraction of data set aside for testing (default: %(default)s)", type=float, default=0.3)
    parser.add_argument("--validation-set-size",
                        help="Size of validation set (taken out of training set) (default: %(default)s)", type=int,
                        default=60)
    parser.add_argument("--num-passes",
                        help="Number of passes over the data during traning (default: %(default)s)", type=int,
                        default=10)
    parser.add_argument("--max-batch-size",
                        help="Maximum batch size used during training (default: %(default)s)", type=int, default=1000)
    parser.add_argument("--subbatch-max-size",
                        help="Maximum batch size used for gradient calculation (default: %(default)s)", type=int,
                        default=25)
    parser.add_argument("--model-checkpoint-path",
                        help="Where to dump/read model checkpoints (default: %(default)s)", default="models")
    parser.add_argument("--max-to-keep",
                        help="Maximal number of checkpoints to keep (default: %(default)s)", type=int, default=5)
    parser.add_argument("--read-from-checkpoint", action="store_true",
                        help="Whether to read model from checkpoint")
    parser.add_argument("--mode", choices=['train', 'test'], default="train",
                        help="Mode of operation (default: %(default)s)")
    parser.add_argument("--model-output-type", choices=['en_class', 'en_val'], default="en_class",
                        help="Whether the model should output dG value or classification labels (default: %(default)s)")
    parser.add_argument("--dropout-keep-prob", type=float, default=0.5,
                        help="Probability for leaving out node in dropout (default: %(default)s)")
    parser.add_argument("--learning-rate",
                        help="Learing rate for Adam (default: %(default)s)", type=float, default=0.001)
    parser.add_argument("--reg-fact",
                        help="Regularisation factor (default: %(default)s)", type=float, default=0.001)
    parser.add_argument("--output-interval",
                        help="The output interval for train and validation error  (default: %(default)s)", type=int,
                        default=20)
    parser.add_argument("--model", choices=list(models.keys()), required=True,
                        help="Which model definition to use (default: %(default)s)")
    parser.add_argument("--step", type=int, default=None,
                        help="Which checkpoint file to use (default: %(default)s)")

    options = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(options).items()):
        print(key, "=", value)

    high_res_protein_feature_filenames = sorted(
        glob.glob(os.path.join(options.high_res_features_input_dir, "*protein_features.npz")))
    high_res_grid_feature_filenames = sorted(
        glob.glob(os.path.join(options.high_res_features_input_dir, "*residue_features.npz")))

    train_start = 0
    validation_end = test_start = 10*int((len(high_res_protein_feature_filenames)//10 * (1. - options.test_set_fraction)))
    train_end = validation_start = int(validation_end - options.validation_set_size)
    test_end = len(high_res_protein_feature_filenames)

    print("# Data:")
    print("Number of unique proteins: ", len(high_res_protein_feature_filenames)//10)
    print("Total size: ", len(high_res_protein_feature_filenames))
    print("Training size: ", train_end - train_start)
    print("Validation size: ", validation_end - validation_start)
    print("Test size: ", test_end - test_start)

    if options.mode == 'train':
        batch_factory = BatchFactory()
        batch_factory.add_data_set("high_res",
                                   high_res_protein_feature_filenames[:train_end],
                                   high_res_grid_feature_filenames[:train_end])
        batch_factory.add_data_set("model_output",
                                   high_res_protein_feature_filenames[:train_end],
                                   key_filter=[options.model_output_type])
        print("Train dataset added")
        validation_batch_factory = BatchFactory()
        validation_batch_factory.add_data_set("high_res",
                                              high_res_protein_feature_filenames[validation_start:validation_end],
                                              high_res_grid_feature_filenames[validation_start:validation_end])
        validation_batch_factory.add_data_set("model_output",
                                              high_res_protein_feature_filenames[validation_start:validation_end],
                                              key_filter=[options.model_output_type])
        print("Validation fraction added")
    elif options.mode == 'test':
        batch_factory = BatchFactory()
        batch_factory.add_data_set("high_res",
                                   high_res_protein_feature_filenames[test_start:],
                                   high_res_grid_feature_filenames[test_start:])
        batch_factory.add_data_set("model_output",
                                   high_res_protein_feature_filenames[test_start:],
                                   key_filter=[options.model_output_type])
    else:
        raise KeyError("Invalid mode")

    holder = batch_factory.next(1, increment_counter=False)
    high_res_grid_size = holder[0]["high_res"].shape
    output_size = batch_factory.next(1, increment_counter=False)[0]["model_output"].shape[1]

    if options.model.startswith("Spherical"):
        model = models[options.model](r_size_high_res=high_res_grid_size[1],
                                      theta_size_high_res=high_res_grid_size[2],
                                      phi_size_high_res=high_res_grid_size[3],
                                      channels_high_res=high_res_grid_size[4],
                                      output_size=output_size,
                                      reg_fact=options.reg_fact,
                                      learning_rate=options.learning_rate,
                                      model_checkpoint_path=options.model_checkpoint_path,
                                      max_to_keep=options.max_to_keep)
    elif options.model.startswith("CubedSphere"):
        model = models[options.model](patches_size_high_res=high_res_grid_size[1],
                                      r_size_high_res=high_res_grid_size[2],
                                      xi_size_high_res=high_res_grid_size[3],
                                      eta_size_high_res=high_res_grid_size[4],
                                      channels_high_res=high_res_grid_size[5],
                                      output_size=output_size,
                                      reg_fact=options.reg_fact,
                                      learning_rate=options.learning_rate,
                                      model_checkpoint_path=options.model_checkpoint_path,
                                      max_to_keep=options.max_to_keep)
    elif options.model.startswith("Cartesian"):
        model = models[options.model](x_size_high_res=high_res_grid_size[1],
                                      y_size_high_res=high_res_grid_size[2],
                                      z_size_high_res=high_res_grid_size[3],
                                      channels_high_res=high_res_grid_size[4],
                                      output_size=output_size,
                                      reg_fact=options.reg_fact,
                                      learning_rate=options.learning_rate,
                                      model_checkpoint_path=options.model_checkpoint_path,
                                      max_to_keep=options.max_to_keep)
    else:
        raise argparse.ArgumentTypeError("Model type not suppported: %s" % options.model)

    if options.read_from_checkpoint:
        model.restore(options.model_checkpoint_path, step=options.step)

    if options.mode == 'train':
        # noinspection PyUnboundLocalVariable
        model.train(train_batch_factory=batch_factory,
                    validation_batch_factory=validation_batch_factory,
                    num_passes=options.num_passes,
                    max_batch_size=options.max_batch_size,
                    subbatch_max_size=options.subbatch_max_size,
                    dropout_keep_prob=options.dropout_keep_prob,
                    output_interval=options.output_interval)

    elif options.mode == 'test':

        prev_pdb_id = None
        pdb_ids = set()
        all_identical = np.array([])
        all_entropies = np.array([])

        more_data = True

        while more_data:
            batch, subbatch_sizes = batch_factory.next(test_end - test_start,
                                                       enforce_protein_boundaries=True,
                                                       include_pdb_ids=True,
                                                       return_single_proteins=True)
            more_data = (batch_factory.feature_index != 0)

            grid_matrix = batch["high_res"]

            vals = batch["model_output"]

            feed_dict = dict({model.x_high_res: grid_matrix,
                              model.y: vals,
                              model.dropout_keep_prob: 1})

            y = vals
            y_argmax = np.argmax(y, 1)
            results = model._infer(batch, np.array([test_end-test_start]), var=[model.layers[-1]['dense'], model.entropy],
                                  include_output=True)
            y_, entropies = list(map(np.concatenate, list(zip(*results))))
            with open("list2.txt", "w") as wfile:
                for line in sorted(high_res_protein_feature_filenames[test_start:]):
                    wfile.write(f"{line}\n")
            # Note that the return_single_proteins make sure that the batch always returns a whole protein
            predictions = np.argmax(y_, 1)

            identical = (predictions == y_argmax)
            Q_accuracy = np.mean(identical)
        colored_vals = [f"{BCOLORS.OKGREEN}{str(elem)}{BCOLORS.ENDC}" if identical[idx] else
                        f"{BCOLORS.WARNING}{str(elem)}{BCOLORS.ENDC}" for idx, elem in enumerate(y_argmax)]
        colored_preds = [f"{BCOLORS.OKGREEN}{str(elem)}{BCOLORS.ENDC}" if identical[idx] else
                         f"{BCOLORS.WARNING}{str(elem)}{BCOLORS.ENDC}" for idx, elem in enumerate(predictions)]
        print("True values:")
        print(*colored_vals, sep=" ")
        print("Prediction:")
        print(*colored_preds, sep=" ")
        print("# Statistics for the whole dataset:")
        # noinspection PyStringFormat
        print("# Q%s score (test set): %f" % (output_size, Q_accuracy))
        print("# loss (test set):", np.mean(entropies) + options.reg_fact)
        F_score = report(y_argmax, predictions, target_names=['Worse', 'Same', 'Better'])
        print(F_score)

