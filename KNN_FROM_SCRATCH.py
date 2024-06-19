#      KNN Algorithm from Scratch by: Charaf Eddine BENARAB


from math import sqrt
import csv
import heapq
from plot_data import plot_data
from typing import List, Tuple,  Dict
from typing_extensions import Literal
from colorama import Fore


# Extracting features and labels from A CSV FILE

def load_data(csv_file_path: str) -> Tuple[List[List[float]], List[str]]:
    """parses and reads data from a csv file and returns a tuple containing a list of data rows and a list of labels."""
    try:
        with open(csv_file_path) as csv_ready:
            csv_reader = csv.reader(csv_ready, delimiter=",")
            labels = []
            row_data = []
            dataset = []
            for row in csv_reader:
                if row:  # to ensure the row is no tempty
                    labels.append(row[-1])
                    row_data = list(map(float, row[:-1]))
                    dataset.append(row_data)
    except FileNotFoundError:
        print(f"{Fore.RED}File {csv_file_path} not found {Fore.RESET}")
    print(f"{Fore.GREEN}File {csv_file_path} loaded successfully {Fore.RESET}")
    return dataset, labels


# Assuming only Euclidean_distance for its Popularity





def calculate_distance(
    dp1: List[float],
    dp2: List[float],
    metric: Literal["euclidean", "manhattan"] = "euclidean",
) -> float:
    if len(dp1) != len(dp2):
        raise ValueError("The two data points must be of the same dimension")

    if metric == "euclidean":
        square_dist = 0.0
        for x, y in zip(dp1, dp2):
            square_dist += (x - y) ** 2
        dist = sqrt(square_dist)
    elif metric == "manhattan":
        dist = 0.0
        for x, y in zip(dp1, dp2):
            dist += abs(x - y)
    else:
        raise ValueError(f"The distance metric '{metric}' is not supported")

    return dist


# Finding Neighbours


def set_elems_dists(
    dataset: List[List[float]], labels: List[str], _input: List[float]
) -> Tuple[List[float], List[int], Dict[float, int], Dict[int, str]]:
    """Calculate the distances of all data points in the dataset from the input point
    and store relevant information."""
    distances = []
    data_elements = []
    data_dict = {}
    id2label = {}
    # Storing Data elements and their Distance from Input element
    for _id, _data in enumerate(dataset):
        distance = calculate_distance(dp1=_data, dp2=_input)
        distances.append(distance)
        id2label[_id] = labels[_id]

        data_elements.append(_id)
        data_dict[distance] = _id

    return distances, data_elements, data_dict, id2label


# Finiding the nearest points to our input
def find_neighbours(
    distances: List[float],
    data_dict: Dict[float, int],
    k: int,
    id2label: Dict[int, str],
) -> Tuple[List[int], List[str]]:
    if k > len(data_dict):
        raise ValueError(
            f"{Fore.RED}k must be less than the number of data points{Fore.RESET}"
        )
    neighbours_ids = []
    neighbours_labels = []

    min_dists = heapq.nsmallest(k, distances)  # get the k smallest distances
    for dist in min_dists:
        neighbours_ids.append(data_dict[dist])
        neighbours_labels.append(id2label[data_dict[dist]])

    return neighbours_ids, neighbours_labels


# FUnction for finding element with majority in a list
def majority_element(num_list: List[int]) -> int:
    """Find the majority element in a list using the Boyer-Moore Voting Algorithm."""
    candidate_idx, counter = 0, 1
    # Find a candidate for the majority element
    for i in range(1, len(num_list)):
        if num_list[candidate_idx] == num_list[i]:
            counter += 1
        else:
            counter -= 1
            if counter == 0:
                idx = i
                counter = 1
    candidate = num_list[candidate_idx]
    # Verify that the candidate is the majority element
    count = sum(1 for num in num_list if num == candidate)
    if count > len(num_list) // 2:
        return candidate
    else:
        raise ValueError("No majority element found.")

    


def KNN_algorithm(
    data_x: List[List[float]], data_y: List[str], _input: List[float], k: int
):
    """K-Nearest Neighbors algorithm to classify the input
    based on the majority label of its k nearest neighbors."""

    if k > len(data_x):
        raise ValueError(
            f"{Fore.RED}k must be less than the number of data points{Fore.RESET}"
        )
    distances, data_elements, data_dict, id2label = set_elems_dists(
        data_x, data_y, _input
    )
    _, neighbours_labels = find_neighbours(
         distances, data_dict, k, id2label
    )

    return majority_element(neighbours_labels)


def get_user_input(num_features: int) -> Tuple[List[float], int]:
    """
    Get input features and the number of neighbors from the user.

    Args:
        num_features (int): Number of features to input.

    Returns:
        Tuple[List[float], int]: A tuple containing the input features and the number of neighbors.
    """
    _input = []
    for i in range(num_features):
        while True:
            try:
                feature = float(input(f"{Fore.CYAN}Enter feature Number {i + 1}: {Fore.RESET}  \n"))
                _input.append(feature)
                break
            except ValueError:
                print(f"{Fore.RED}Invalid input. Please enter a valid number.{Fore.RESET}")

    while True:
        try:
            k = int(input("Enter number of Neighbors:   "))
            if k <= 0:
                raise ValueError(
                    f"{Fore.RED}Number of neighbors must be a positive integer.{Fore.RESET}"
                )
            break
        except ValueError as e:
            print(e)
    return _input, k


def main():
    data_x, data_y = load_data("iris.csv")
    plot_data(data_x)
    while True:
        try:
            _input, k = get_user_input(num_features=4)
            category = KNN_algorithm(data_x=data_x, data_y=data_y, _input=_input, k=k)
            print(f"{Fore.MAGENTA}The predicted category is: {category}{Fore.RESET}")
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":
    main()
