# Copyright (C) 2017-2023  Cleanlab Inc.
# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with cleanlab.  If not, see <https://www.gnu.org/licenses/>.

import requests
import pytest
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import io
import numpy as np
from hypothesis import given, settings
from cleanlab.dataset import (
    health_summary,
    find_overlapping_classes,
    rank_classes_by_label_quality,
    overall_label_health_score,
)
from cleanlab.count import estimate_joint, num_label_issues, compute_confident_joint

cifar100 = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]
caltech256 = [
    "ak47",
    "american-flag",
    "backpack",
    "baseball-bat",
    "baseball-glove",
    "basketball-hoop",
    "bat",
    "bathtub",
    "bear",
    "beer-mug",
    "billiards",
    "binoculars",
    "birdbath",
    "blimp",
    "bonsai",
    "boom-box",
    "bowling-ball",
    "bowling-pin",
    "boxing-glove",
    "brain",
    "breadmaker",
    "buddha",
    "bulldozer",
    "butterfly",
    "cactus",
    "cake",
    "calculator",
    "camel",
    "cannon",
    "canoe",
    "car-tire",
    "cartman",
    "cd",
    "centipede",
    "cereal-box",
    "chandelier",
    "chess-board",
    "chimp",
    "chopsticks",
    "cockroach",
    "coffee-mug",
    "coffin",
    "coin",
    "comet",
    "computer-keyboard",
    "computer-monitor",
    "computer-mouse",
    "conch",
    "cormorant",
    "covered-wagon",
    "cowboy-hat",
    "crab",
    "desk-globe",
    "diamond-ring",
    "dice",
    "dog",
    "dolphin",
    "doorknob",
    "drinking-straw",
    "duck",
    "dumb-bell",
    "eiffel-tower",
    "electric-guitar",
    "elephant",
    "elk",
    "ewer",
    "eyeglasses",
    "fern",
    "fighter-jet",
    "fire-extinguisher",
    "fire-hydrant",
    "fire-truck",
    "fireworks",
    "flashlight",
    "floppy-disk",
    "football-helmet",
    "french-horn",
    "fried-egg",
    "frisbee",
    "frog",
    "frying-pan",
    "galaxy",
    "gas-pump",
    "giraffe",
    "goat",
    "golden-gate-bridge",
    "goldfish",
    "golf-ball",
    "goose",
    "gorilla",
    "grand-piano",
    "grapes",
    "grasshopper",
    "guitar-pick",
    "hamburger",
    "hammock",
    "harmonica",
    "harp",
    "harpsichord",
    "hawksbill",
    "head-phones",
    "helicopter",
    "hibiscus",
    "homer-simpson",
    "horse",
    "horseshoe-crab",
    "hot-air-balloon",
    "hot-dog",
    "hot-tub",
    "hourglass",
    "house-fly",
    "human-skeleton",
    "hummingbird",
    "ibis",
    "ice-cream-cone",
    "iguana",
    "ipod",
    "iris",
    "jesus-christ",
    "joy-stick",
    "kangaroo",
    "kayak",
    "ketch",
    "killer-whale",
    "knife",
    "ladder",
    "laptop",
    "lathe",
    "leopards",
    "license-plate",
    "lightbulb",
    "light-house",
    "lightning",
    "llama",
    "mailbox",
    "mandolin",
    "mars",
    "mattress",
    "megaphone",
    "menorah",
    "microscope",
    "microwave",
    "minaret",
    "minotaur",
    "motorbikes",
    "mountain-bike",
    "mushroom",
    "mussels",
    "necktie",
    "octopus",
    "ostrich",
    "owl",
    "palm-pilot",
    "palm-tree",
    "paperclip",
    "paper-shredder",
    "pci-card",
    "penguin",
    "people",
    "pez-dispenser",
    "photocopier",
    "picnic-table",
    "playing-card",
    "porcupine",
    "pram",
    "praying-mantis",
    "pyramid",
    "raccoon",
    "radio-telescope",
    "rainbow",
    "refrigerator",
    "revolver",
    "rifle",
    "rotary-phone",
    "roulette-wheel",
    "saddle",
    "saturn",
    "school-bus",
    "scorpion",
    "screwdriver",
    "segway",
    "self-propelled-lawn-mower",
    "sextant",
    "sheet-music",
    "skateboard",
    "skunk",
    "skyscraper",
    "smokestack",
    "snail",
    "snake",
    "sneaker",
    "snowmobile",
    "soccer-ball",
    "socks",
    "soda-can",
    "spaghetti",
    "speed-boat",
    "spider",
    "spoon",
    "stained-glass",
    "starfish",
    "steering-wheel",
    "stirrups",
    "sunflower",
    "superman",
    "sushi",
    "swan",
    "swiss-army-knife",
    "sword",
    "syringe",
    "tambourine",
    "teapot",
    "teddy-bear",
    "teepee",
    "telephone-box",
    "tennis-ball",
    "tennis-court",
    "tennis-racket",
    "theodolite",
    "toaster",
    "tomato",
    "tombstone",
    "top-hat",
    "touring-bike",
    "tower-pisa",
    "traffic-light",
    "treadmill",
    "triceratops",
    "tricycle",
    "trilobite",
    "tripod",
    "t-shirt",
    "tuning-fork",
    "tweezer",
    "umbrella",
    "unicorn",
    "vcr",
    "video-projector",
    "washing-machine",
    "watch",
    "waterfall",
    "watermelon",
    "welding-mask",
    "wheelbarrow",
    "windmill",
    "wine-bottle",
    "xylophone",
    "yarmulke",
    "yo-yo",
    "zebra",
    "airplanes",
    "car-side",
    "faces-easy",
    "greyhound",
    "tennis-shoes",
    "toad",
]
imdb = ["Negative", "Positive"]
mnist = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

urls = {
    "caltech256": [
        "https://github.com/cleanlab/label-errors/raw/5392f6c71473055060be3044becdde1cbc18284d/"
        "original_test_labels/caltech256_original_labels.npy",
        "https://github.com/cleanlab/label-errors/raw/5392f6c71473055060be3044becdde1cbc18284d"
        "/cross_validated_predicted_probabilities/caltech256_pyx.npy",
    ],
    "mnist": [
        "https://github.com/cleanlab/label-errors/raw/5392f6c71473055060be3044becdde1cbc18284d"
        "/original_test_labels/mnist_test_set_original_labels.npy",
        "https://github.com/cleanlab/label-errors/raw/5392f6c71473055060be3044becdde1cbc18284d"
        "/cross_validated_predicted_probabilities/mnist_test_set_pyx.npy",
    ],
    "imdb": [
        "https://github.com/cleanlab/label-errors/raw"
        "/5392f6c71473055060be3044becdde1cbc18284d/original_test_labels"
        "/imdb_test_set_original_labels.npy",
        "https://github.com/cleanlab/label-errors/raw/5392f6c71473055060be3044becdde1cbc18284d"
        "/cross_validated_predicted_probabilities/imdb_test_set_pyx.npy",
    ],
    "cifar100": [
        "https://github.com/cleanlab/label-errors/raw/5392f6c71473055060be3044becdde1cbc18284d"
        "/original_test_labels/cifar100_test_set_original_labels.npy",
        "https://github.com/cleanlab/label-errors/raw/5392f6c71473055060be3044becdde1cbc18284d"
        "/cross_validated_predicted_probabilities/cifar100_test_set_pyx.npy",
    ],
}


def _get_pred_probs_labels_from_labelerrors_datasets(dataset_name):
    """Helper function to load data from the labelerrors.com datasets."""

    labels_url, pred_probs_url = urls[dataset_name]
    response = requests.get(pred_probs_url)
    response.raise_for_status()
    pred_probs = np.load(io.BytesIO(response.content), allow_pickle=True)
    response = requests.get(labels_url)
    response.raise_for_status()
    labels = np.load(io.BytesIO(response.content), allow_pickle=True)
    return pred_probs, labels


@pytest.mark.parametrize("dataset_name", ["mnist", "caltech256", "cifar100"])
def test_real_datasets(dataset_name):
    print("\n" + dataset_name.capitalize() + "\n")
    class_names = eval(dataset_name)
    pred_probs, labels = _get_pred_probs_labels_from_labelerrors_datasets(dataset_name)
    # if this runs without issue no all four datasets, the test passes
    _ = health_summary(
        pred_probs=pred_probs,
        labels=labels,
        class_names=class_names,
        verbose=dataset_name != "mnist",  # test out verbose=False on one of the datasets.
    )


@pytest.mark.parametrize("dataset_name", ["mnist"])
def test_multilabel_error(dataset_name):
    print("\n" + dataset_name.capitalize() + "\n")
    class_names = eval(dataset_name)
    pred_probs, labels = _get_pred_probs_labels_from_labelerrors_datasets(dataset_name)
    # if this runs without issue no all four datasets, the test passes
    with pytest.raises(ValueError) as e:
        _ = find_overlapping_classes(labels=labels, pred_probs=pred_probs, multi_label=True)


@pytest.mark.parametrize("asymmetric", [True, False])
@pytest.mark.parametrize("dataset_name", ["mnist", "imdb"])
def test_symmetry_df_size(asymmetric, dataset_name):
    pred_probs, labels = _get_pred_probs_labels_from_labelerrors_datasets(dataset_name)
    joint = estimate_joint(labels=labels, pred_probs=pred_probs)
    num_classes = pred_probs.shape[1]
    df = find_overlapping_classes(
        joint=joint,
        asymmetric=asymmetric,
        class_names=eval(dataset_name),
        num_examples=len(labels),
    )
    if asymmetric:
        assert len(df) == num_classes**2 - num_classes
    else:  # symmetric
        assert len(df) == (num_classes**2 - num_classes) / 2

        # Second test for symmetric
        # check that the row, col value returned is actually the sum from the joint.
        sum_0_1 = joint[0, 1] + joint[1, 0]
        df_0_1 = df[(df["Class Index A"] == 0) & (df["Class Index B"] == 1)]["Joint Probability"]
        assert sum_0_1 - df_0_1.values[0] < 1e-8  # Check two floats are equal


@pytest.mark.parametrize("use_num_examples", [True, False])
@pytest.mark.parametrize("use_labels", [True, False])
@pytest.mark.parametrize(
    "func", [find_overlapping_classes, rank_classes_by_label_quality, overall_label_health_score]
)
def test_value_error_missing_num_examples_with_joint(use_num_examples, use_labels, func):
    dataset_name = "imdb"
    pred_probs, labels = _get_pred_probs_labels_from_labelerrors_datasets(dataset_name)
    joint = estimate_joint(labels=labels, pred_probs=pred_probs)
    if use_num_examples is False and use_labels is False:  # can't infer num_examples. Throw error!
        with pytest.raises(ValueError) as e:
            df = func(
                labels=labels if use_labels else None,
                joint=joint,
                num_examples=len(labels) if use_num_examples else None,
            )
    else:  # at least one of use_num_examples and use_labels must be True. Can infer num_examples.
        # If this runs without error, the test passes.
        df = func(
            labels=labels if use_labels else None,
            joint=joint,
            num_examples=len(labels) if use_num_examples else None,
        )


@pytest.mark.parametrize("dataset_name", ["mnist", "caltech256", "cifar100"])
def test_overall_label_health_score_matched_num_issues(dataset_name):
    # Matches num_label_issues
    pred_probs, labels = _get_pred_probs_labels_from_labelerrors_datasets(dataset_name)
    num_issues = num_label_issues(labels=labels, pred_probs=pred_probs)
    score = overall_label_health_score(labels=labels, pred_probs=pred_probs)
    assert 1 - num_issues / labels.shape[0] == score


def test_overall_label_health_score_function_calls():
    dataset_name = "caltech256"
    pred_probs, labels = _get_pred_probs_labels_from_labelerrors_datasets(dataset_name)
    score = overall_label_health_score(labels=labels, pred_probs=pred_probs)

    confident_joint = compute_confident_joint(labels=labels, pred_probs=pred_probs)
    num_examples = len(labels)
    score_cj = overall_label_health_score(
        labels=None, pred_probs=pred_probs, confident_joint=confident_joint
    )
    joint = estimate_joint(labels=labels, pred_probs=pred_probs)
    score_joint = overall_label_health_score(
        labels=None, pred_probs=pred_probs, joint=joint, num_examples=num_examples
    )
    joint_cj = estimate_joint(labels=labels, pred_probs=pred_probs, confident_joint=confident_joint)
    score_joint_cj = overall_label_health_score(
        labels=None, pred_probs=pred_probs, joint=joint_cj, num_examples=num_examples
    )
    assert score_cj != score
    assert score_cj == score_joint
    assert score_joint_cj == score_joint


confident_joint_strategy = npst.arrays(
    np.int32,
    shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10),
    elements=st.integers(min_value=0, max_value=int(1e6)),
).filter(lambda arr: arr.shape[0] == arr.shape[1])


@pytest.mark.issue_651
@given(confident_joint=confident_joint_strategy)
@settings(deadline=500)
def test_find_overlapping_classes_with_confident_joint(confident_joint):
    # Setup
    K = confident_joint.shape[0]
    overlapping_classes = find_overlapping_classes(confident_joint=confident_joint)

    # Test that the output dataframe has the expected columns
    expected_columns = [
        "Class Index A",
        "Class Index B",
        "Num Overlapping Examples",
        "Joint Probability",
    ]
    assert set(overlapping_classes.columns) == set(expected_columns)

    # Class indices must be valid
    assert overlapping_classes["Class Index A"].between(0, K - 1).all()
    assert overlapping_classes["Class Index B"].between(0, K - 1).all()

    # Overlapping example count should be non-negative integers
    assert (overlapping_classes["Num Overlapping Examples"] >= 0).all()
    assert overlapping_classes["Num Overlapping Examples"].dtype == int

    # Joint probabilities should be between 0 and 1
    assert (overlapping_classes["Joint Probability"] >= 0).all()
    assert (overlapping_classes["Joint Probability"] <= 1).all()

    # Joint probabilities sorted in descending order
    if K > 2:
        assert (overlapping_classes["Joint Probability"].diff()[1:] <= 0).all()
