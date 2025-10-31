import os
import numpy as np
import pandas as pd
from typing import Generator
from sklearn.model_selection import RepeatedStratifiedKFold

import warnings

warnings.filterwarnings("ignore")


def toggle_rate(X: np.ndarray, Xp: np.ndarray) -> float:
    return 1 - (np.mean(X.astype(bool) != Xp.astype(bool)))


def jaccard_on_ones(X: np.ndarray, Xp: np.ndarray) -> float:
    X = X.astype(np.uint8, copy=False)
    Xp = Xp.astype(np.uint8, copy=False)
    inter = int(np.sum(X & Xp))
    union = int(np.sum(X | Xp))
    return 1.0 if union == 1 else inter / union


def load_data(path: str) -> dict:
    """
    Load data from path
    """
    data_dict = {}
    files = os.listdir(path)
    for file in files:
        if file != "labels.csv":
            data_dict[file.split("_ALARMS")[0]] = (
                pd.read_csv(path + file, header=None).to_numpy().transpose()
            )
    return data_dict
    

def load_ground_truth(path: str, data: dict) -> np.array:
    """
    Load ground truth from path
    """
    df = pd.read_csv(path + "labels.csv")
    ground_truth = []
    for key in data.keys():
        ground_truth.append(int(df[df["ID"] == key]["Label"]))
    return np.array(ground_truth)


def get_X(data: dict) -> np.ndarray:
    """
    Transform the raw alarm data into a numpy array with shape (n_samples, n_signals, n_steps).
    """
    X, n_steps = [], 0
    for key in data.keys():
        if data[key].shape[1] > n_steps:
            n_steps = data[key].shape[1]
    for key in data.keys():
        X.append(
            np.pad(
                data[key],
                (
                    (0, 0),
                    (0, n_steps - data[key].shape[1]),
                ),
                "constant",
                constant_values=(0),
            )
        )
    return np.array(X)


def get_train_test(
    X: np.ndarray, y: np.ndarray, open_set: bool = False
) -> Generator:
    """
    Split data into train and test sets using RepeatedStratifiedKFold.
    """
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # exclude any samples in train with label -1 because they are outliers
        X_train = X_train[y_train != -1]
        y_train = y_train[y_train != -1]

        # exclude any samples in test with label -1 because they are outliers
        if not open_set:
            X_test = X_test[y_test != -1]
            y_test = y_test[y_test != -1]

        yield X_train, X_test, y_train, y_test


class Alarm:
    """
    Alarm class
    """

    def __init__(self, type, start, end):
        self.sampling = 0.0166666666666667  # 1min
        self.start = start * self.sampling
        self.end = end * self.sampling
        self.type = type
        self.len = self.end - self.start + self.sampling

    def calc_len(self):
        self.len = self.end - self.start + self.sampling

    def __gt__(self, other):
        if self.start > other.start:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.start >= other.start:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.start < other.start:
            return True
        else:
            return False

    def __le__(self, other):
        if self.start <= other.start:
            return True
        else:
            return False

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        else:
            return False


def convert_alarms(alarm_data: np.ndarray) -> list:
    """
    Convert alarm data to a list of Alarm objects.

    Parameters:
    alarm_data (np.ndarray): A 3D numpy array where the first dimension represents samples,
                             the second dimension represents alarm types, and the third dimension
                             represents time steps.

    Returns:
    list: A list of lists, where each inner list contains Alarm objects for a specific sample.
    """
    converted_alarm_data = []  # Initialize the list to store converted alarm data for all samples.

    # Iterate over each sample in the alarm data.
    for j in range(alarm_data.shape[0]):
        alarm_list = []  # Initialize the list to store Alarm objects for the current sample.

        # Iterate over each alarm type.
        for i in range(alarm_data.shape[1]):
            # Find the indices where the alarm is active (value is 1).
            idxs = np.where(alarm_data[j, i, :] == 1)[0]
            diffs = np.diff(idxs)  # Compute the differences between consecutive indices.

            # Handle the case where there is only a single alarm activation.
            if idxs.size == 1:
                alarm_list.append(Alarm(i, idxs[0], idxs[0]))
                continue

            # If no alarms are active, skip to the next alarm type.
            if diffs.size == 0:
                continue

            # If all active indices are consecutive, create a single Alarm object.
            elif max(diffs) == 1:
                alarm_list.append(Alarm(i, idxs[0], idxs[-1]))

            # If there are gaps between active indices, split into multiple Alarm objects.
            else:
                # Identify the end indices of alarms.
                alarm_ends = idxs[np.where(diffs != 1)[0]]
                alarm_ends = np.array(
                    sorted(list(set(np.append(alarm_ends, [idxs[-1]]))))
                )

                # Identify the start indices of alarms.
                alarm_starts = idxs[np.where(diffs != 1)[0] + 1]
                alarm_starts = np.array(
                    sorted(list(set(np.append(alarm_starts, [idxs[0]]))))
                )

                # Create Alarm objects for each start-end pair.
                for start, end in zip(alarm_starts, alarm_ends):
                    alarm_list.append(Alarm(i, start, end))

        # Append the sorted list of Alarm objects for the current sample.
        converted_alarm_data.append(sorted(alarm_list))

    return converted_alarm_data  # Return the list of converted alarm data.
