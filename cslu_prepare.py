"""
Data preparation for CSLU.

Authors
* Peter Plantinga 2020
"""

import os
import re
import csv
import logging
from glob import glob
from speechbrain.utils.data_utils import get_all_files
from speechbrain.data_io.data_io import read_wav_soundfile

logger = logging.getLogger(__name__)
TRAIN_CSV = "train.csv"
DEV_CSV = "dev.csv"
TEST_CSV = "test.csv"
SAMPLERATE = 16000


def prepare_cslu(
    data_folder,
    save_folder,
    dev_frac=0.05,
    test_frac=0.05,
):
    """
    Prepares the csv files for the CSLU dataset.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original CSLU dataset is stored.
    save_folder : str
        Path to the folder where prepared csv files are stored.
    dev_frac : float
        Approximate fraction of speakers to assign to dev data in each grade.
    test_frac : float
        Approximate fraction of speakers to assign to test data in each grade.
    """
    # Setting file extension.
    extension = [".wav"]

    # Setting the save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Setting ouput files
    save_csv_train = os.path.join(save_folder, TRAIN_CSV)
    save_csv_dev = os.path.join(save_folder, DEV_CSV)
    save_csv_test = os.path.join(save_folder, TEST_CSV)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder):
        logger.debug("Skipping preparation, completed in previous run.")
        return

    msg = "\tCreating csv file for the CSLU Dataset.."
    logger.debug(msg)

    speech_dir = os.path.join(data_folder, "speech", "scripted")
    grade_list = [
        "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"
    ]

    # Creating csv file for training data
    wav_lst_train = []
    wav_lst_dev = []
    wav_lst_test = []
    for grade in grade_list:
        grade_dir = os.path.join(speech_dir, grade)

        # Collect list of folders for distinct speakers
        spk_list = sorted(glob(os.path.join(grade_dir, "*", "ks*")))

        # Divide into sublists
        dev_len = int(dev_frac * len(spk_list))
        test_len = int(test_frac * len(spk_list))
        dev_spk_list = spk_list[:dev_len]
        test_spk_list = spk_list[dev_len:dev_len + test_len]
        train_spk_list = spk_list[dev_len + test_len:]

        # Find files and add em
        for folder in train_spk_list:
            wav_lst_train.extend(get_all_files(folder, match_and=extension))
        for folder in dev_spk_list:
            wav_lst_dev.extend(get_all_files(folder, match_and=extension))
        for folder in test_spk_list:
            wav_lst_test.extend(get_all_files(folder, match_and=extension))

    # Create data maps
    id2chars = load_id2chars_map(data_folder)
    id2verify = load_id2verify_map(data_folder, grade_list)

    # Create csv with all files
    create_csv(wav_lst_train, save_csv_train, id2chars, id2verify)
    create_csv(wav_lst_dev, save_csv_dev, id2chars, id2verify)
    create_csv(wav_lst_test, save_csv_test, id2chars, id2verify)

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
    # Checking csv files
    skip = True

    for csv in [TRAIN_CSV, DEV_CSV, TEST_CSV]:
        if not os.path.isfile(os.path.join(save_folder, csv)):
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
        words = re.sub(r'[^a-z ]', '', words).replace(" ", "_")

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
            id2verify[utterance_id] = verify

    return id2verify

def create_csv(wav_lst, csv_file, id2chars, id2verify):
    """
    Creates the csv file given a list of wav files.

    Arguments
    ---------
    wav_lst : list
        The list of wav files of a given data split.
    csv_file : str
        The path of the output csv file
    id2chars : dict
        A mapping from word id to characters
    id2verify : dict
        A mapping from utterance id to verify label
    """

    # Adding some Prints
    msg = '\t"Creating csv lists in  %s..."' % (csv_file)
    logger.debug(msg)

    csv_lines = [
        [
            "ID",
            "duration",
            "wav",
            "wav_format",
            "wav_opts",
            "char",
            "char_format",
            "char_opts",
            "verify",
            "verify_format",
            "verify_opts",
        ]
    ]

    # Processing all the wav files in the list
    for wav_file in wav_lst:

        # Reading the signal (to retrieve duration in seconds)
        signal = read_wav_soundfile(wav_file)
        duration = str(signal.shape[0] / SAMPLERATE)

        # Find character targets
        snt_id = wav_file[-12:-4]
        word_id = wav_file[-7:-5].upper()
        chars = id2chars[word_id]

        # Composition of the csv_line
        csv_line = [
            snt_id, duration, wav_file, "wav", "", chars, "string", "",
        ]

        # Verify = 0 means there is no label for this example
        #verify = "0" if snt_id not in id2verify else id2verify[snt_id]
        if snt_id not in id2verify:
            continue
        else:
            csv_line.extend([id2verify[snt_id], "string", "label:False"])
            csv_lines.append(csv_line)

    # Writing the csv lines
    _write_csv(csv_lines, csv_file)
    msg = "\t%s sucessfully created!" % (csv_file)
    logger.debug(msg)

def _write_csv(csv_lines, csv_file):
    """
    Writes on the specified csv_file the given csv_files.
    """
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)
