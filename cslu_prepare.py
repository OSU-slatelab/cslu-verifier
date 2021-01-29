"""
Data preparation for CSLU.

Authors
* Peter Plantinga 2020
"""

import os
import re
import json
import logging
from glob import glob
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
TRAIN_FILE = "train.json"
VALID_FILE = "valid.json"
TEST_FILE = "test.json"
SAMPLERATE = 16000


def prepare_cslu(
    data_folder, save_folder, valid_frac=0.05, test_frac=0.05,
):
    """
    Prepares the json files for the CSLU dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original CSLU dataset is stored.
    save_folder : str
        Path to the folder where prepared json files are stored.
    valid_frac : float
        Approximate fraction of speakers to assign to valid data in each grade.
    test_frac : float
        Approximate fraction of speakers to assign to test data in each grade.
    """
    # Setting file extension.
    extension = [".wav"]

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_train_file = os.path.join(save_folder, TRAIN_FILE)
    save_valid_file = os.path.join(save_folder, VALID_FILE)
    save_test_file = os.path.join(save_folder, TEST_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    msg = "\tCreating json file for the CSLU Dataset.."
    logger.debug(msg)

    speech_dir = os.path.join(data_folder, "speech", "scripted")
    grade_list = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    # Creating json file for training data
    wav_lst_train = []
    wav_lst_valid = []
    wav_lst_test = []
    for grade in grade_list:
        grade_dir = os.path.join(speech_dir, grade)

        # Collect list of folders for distinct speakers
        spk_list = sorted(glob(os.path.join(grade_dir, "*", "ks*")))

        # Divide into sublists
        valid_len = int(valid_frac * len(spk_list))
        test_len = int(test_frac * len(spk_list))
        valid_spk_list = spk_list[:valid_len]
        test_spk_list = spk_list[valid_len : valid_len + test_len]
        train_spk_list = spk_list[valid_len + test_len :]

        # Find files and add em
        for folder in train_spk_list:
            wav_lst_train.extend(get_all_files(folder, match_and=extension))
        for folder in valid_spk_list:
            wav_lst_valid.extend(get_all_files(folder, match_and=extension))
        for folder in test_spk_list:
            wav_lst_test.extend(get_all_files(folder, match_and=extension))

    # Create data maps
    id2chars = load_id2chars_map(data_folder)
    id2verify = load_id2verify_map(data_folder, grade_list)

    # Create json with all files
    create_json(wav_lst_train, save_train_file, id2chars, id2verify)
    create_json(wav_lst_valid, save_valid_file, id2chars, id2verify)
    create_json(wav_lst_test, save_test_file, id2chars, id2verify)


def skip(save_folder):
    """
    Detects if the data_preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    # Checking json files
    skip = True

    for json in [TRAIN_FILE, VALID_FILE, TEST_FILE]:
        if not os.path.isfile(os.path.join(save_folder, json)):
            skip = False

    return skip


def load_id2chars_map(data_folder):
    """Load map from word id to characters"""
    word_file = os.path.join(data_folder, "docs", "all.map")
    id2chars = {}
    for line in open(word_file):
        line = line.split()
        if len(line) == 0:
            continue
        word_id = line[0]

        # Normalize
        words = " ".join(line[1:]).lower()
        words = re.sub(r"[^a-z ]", "", words).replace(" ", "_")

        # Split into chars, merging double letters.
        chars = []
        for char in words:
            if len(chars) > 0 and chars[-1] == char:
                chars[-1] = char + char
            else:
                chars.append(char)

        # Store result in map
        id2chars[word_id] = " ".join(chars)

    return id2chars


def load_id2verify_map(data_folder, grades):
    id2verify = {}
    for grade in grades:
        verify_file = os.path.join(data_folder, "docs", grade + "-verified.txt")
        for line in open(verify_file):
            filename, verify = line.split()
            utterance_id = filename[-12:-4]
            id2verify[utterance_id] = int(verify)

    return id2verify


def create_json(wav_lst, json_file, id2chars, id2verify):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    json_file : str
        The path of the output json file
    id2chars : dict
        A mapping from word id to characters
    id2verify : dict
        A mapping from utterance id to verify label
    """

    # Adding some Prints
    msg = '\t"Creating json lists in  %s..."' % (json_file)
    logger.debug(msg)

    json_dict = {}

    # Processing all the wav files in the list
    for wav_file in wav_lst:

        path = os.path.normpath(wav_file).split(os.path.sep)
        relative_filename = os.path.join("{root}", *path[-6:])

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Find character targets
        snt_id = wav_file[-12:-4]
        word_id = wav_file[-7:-5].upper()
        chars = id2chars[word_id]

        # Verify = 0 means there is no label for this example
        # verify = "0" if snt_id not in id2verify else id2verify[snt_id]
        if snt_id not in id2verify:
            continue
        else:
            json_dict[snt_id] = {
                "wav": relative_filename,
                "length": duration,
                "char": chars,
                "verify": id2verify[snt_id],
            }

    # Writing the json lines
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=1)

    msg = "\t%s sucessfully created!" % (json_file)
    logger.debug(msg)
