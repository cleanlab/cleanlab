import pytest
from cleanlab.datalab.data import Data
from datasets import Dataset, load_dataset, ClassLabel
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume


NUM_COLS = 2


@st.composite
def dataset_strategy(draw):
    # Deifine strategies
    int_feature_strategy = st.integers(min_value=-10, max_value=10)
    float_feature_strategy = st.floats(min_value=-10, max_value=10)
    column_name_strategy = st.text(min_size=5, max_size=5)
    column_data_strategy = st.one_of(int_feature_strategy, float_feature_strategy)

    # Draw values
    col_names = draw(
        st.lists(column_name_strategy, min_size=NUM_COLS, max_size=NUM_COLS + 1, unique=True)
    )
    label_name = draw(st.sampled_from(col_names))
    data = {
        name: draw(st.lists(column_data_strategy, min_size=5, max_size=5)) for name in col_names
    }
    dataset = Dataset.from_dict(data)
    dataset = dataset.rename_column(label_name, "label")

    # Make assertions about drawn values
    assume(len(set(dataset["label"])) > 1)

    return dataset


class TestData:
    @pytest.fixture(
        scope="class",
        params=[
            {"name": "lhoestq/demo1", "label_name": "star"},
            {"name": "beans", "label_name": "labels"},
        ],
        autouse=True,
        ids=["demo1", "beans"],
    )
    def dataset_and_label_name(self, request):
        name = request.param["name"]
        split = request.param.get("split", "train")
        label_name = request.param["label_name"]
        dataset = load_dataset(name, split=split)
        yield dataset, label_name

    @pytest.mark.slow
    @given(dataset=dataset_strategy())
    def test_init_data_properties(self, dataset):
        data = Data(data=dataset, label_name="label")
        assert data._data == dataset

        # All elements in the _labels attribute are integers in the range [0, num_classes - 1]
        num_classes = len(set(data._label_map))
        all_labels_are_ints = np.issubdtype(data._labels.dtype, np.integer)
        assert all_labels_are_ints, f"{data._labels} should be a list of integers"
        assert all(0 <= label < num_classes for label in data._labels)

        assert all(isinstance(label, int) for label in data._label_map.keys())

    def test_init_data(self, dataset_and_label_name):
        dataset, label_name = dataset_and_label_name
        data = Data(data=dataset, label_name=label_name)

        label_feature = dataset.features[label_name]
        if isinstance(label_feature, ClassLabel):
            classes = label_feature.names
        else:
            classes = sorted(dataset.unique(label_name))
        assert data.class_names == classes
