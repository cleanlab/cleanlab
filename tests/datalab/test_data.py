import tempfile
from unittest.mock import patch
import pytest
from cleanlab.datalab.internal.data import Data, DataFormatError, DatasetLoadError, LabelNanError
from datasets import Dataset, ClassLabel
import numpy as np
import hypothesis.strategies as st
from hypothesis import given, assume, settings, HealthCheck

from cleanlab.datalab.internal.task import Task


NUM_COLS = 2


@st.composite
def multiclass_dataset_strategy(draw):
    # Define strategies
    int_feature_strategy = st.integers(min_value=-10, max_value=10)
    float_feature_strategy = st.floats(min_value=-10, max_value=10)
    column_name_strategy = st.text(
        alphabet=st.characters(blacklist_categories=["Cs", "Cc", "Cn"]), min_size=5, max_size=5
    )
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


@st.composite
def multilabel_dataset_strategy(draw):
    # Define strategies
    min_dataset_size = 5
    max_dataset_size = 5
    int_feature_strategy = st.integers(min_value=-10, max_value=10)
    float_feature_strategy = st.floats(min_value=-10, max_value=10)
    # Ensure column names do not include problematic characters
    column_name_strategy = st.text(
        alphabet=st.characters(blacklist_characters="\x00", min_codepoint=32, max_codepoint=126),
        min_size=5,
        max_size=5,
    )
    column_data_strategy = st.one_of(int_feature_strategy, float_feature_strategy)

    # Draw values
    col_names = draw(
        st.lists(column_name_strategy, min_size=NUM_COLS, max_size=NUM_COLS + 1, unique=True)
    )
    label_name = draw(st.sampled_from(col_names))
    # Ensure labels do not include problematic characters
    classes_strategy = st.lists(
        st.text(
            alphabet=st.characters(
                # The null character is problematic for some string operations, e.g. key lookup
                blacklist_characters="\x00",
                min_codepoint=32,
                max_codepoint=126,
            ),
            min_size=2,
            max_size=3,
        ),
        min_size=2,
        max_size=3,
        unique=True,
    )
    classes = draw(classes_strategy)
    labels_strategy = st.lists(
        st.lists(st.sampled_from(classes), min_size=1, max_size=3, unique=True),
        min_size=min_dataset_size,
        max_size=max_dataset_size,
    )
    data = {
        name: draw(
            st.lists(column_data_strategy, min_size=min_dataset_size, max_size=max_dataset_size)
        )
        for name in col_names
    }
    data[label_name] = draw(labels_strategy)
    dataset = Dataset.from_dict(data)
    dataset = dataset.rename_column(label_name, "label")

    # Make assertions about drawn values
    assume(len(set(l for labels in dataset["label"] for l in labels)) > 1)

    return dataset


@st.composite
def dataset_strategy(draw, task=Task.CLASSIFICATION):
    if task == Task.CLASSIFICATION:
        return draw(multiclass_dataset_strategy())
    elif task == Task.MULTILABEL:
        return draw(multilabel_dataset_strategy())
    else:
        raise ValueError(f"Unsupported task: {task}")


class TestData:
    @pytest.fixture
    def dataset_and_label_name(self):
        label_name = "labels"

        dataset = Dataset.from_dict({"image": [1, 2, 3], label_name: [0, 1, 0]})
        return dataset, label_name

    @given(dataset=dataset_strategy())
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_init_data_properties(self, dataset):
        data = Data(data=dataset, task=Task.CLASSIFICATION, label_name="label")
        assert data._data == dataset

        # All elements in the _labels attribute are integers in the range [0, num_classes - 1]
        num_classes = len(set(data.labels.label_map))
        all_labels_are_ints = np.issubdtype(data.labels.labels.dtype, np.integer)
        assert all_labels_are_ints, f"{data.labels.labels} should be a list of integers"
        assert all(0 <= label < num_classes for label in data.labels.labels)

        assert all(isinstance(label, int) for label in data.labels.label_map.keys())

    def test_init_data(self, dataset_and_label_name):
        dataset, label_name = dataset_and_label_name
        data = Data(data=dataset, task=Task.CLASSIFICATION, label_name=label_name)

        label_feature = dataset.features[label_name]
        if isinstance(label_feature, ClassLabel):
            classes = label_feature.names
        else:
            classes = sorted(dataset.unique(label_name))
        assert data.class_names == classes

    def test_init_data_from_list_of_dicts(self):
        dataset = [{"X": 0, "label": 0}, {"X": 1, "label": 1}, {"X": 2, "label": 1}]
        data = Data(data=dataset, task=Task.CLASSIFICATION, label_name="label")
        assert isinstance(data._data, Dataset)

    def test_init_raises_format_error(self):
        data = np.random.rand(10, 2)
        with pytest.raises(DataFormatError) as excinfo:
            Data(data=data, task=Task.CLASSIFICATION, label_name="label")

        expected_error_substring = "Unsupported data type: <class 'numpy.ndarray'>\n"
        assert expected_error_substring in str(excinfo.value)

    def test_init_raises_load_error(self):
        improperly_aligned_data = {
            "X": [0, 1, 2],
            "label": [0, 1],
        }
        with pytest.raises(DatasetLoadError) as excinfo:
            Data(data=improperly_aligned_data, task=Task.CLASSIFICATION, label_name="label")

        expected_error_substring = "Failed to load dataset from <class 'dict'>.\n"
        assert expected_error_substring in str(excinfo.value)

    def test_not_equal_to_copy_or_non_data(self):
        dataset = {"X": [0, 1, 2], "label": [0, 1, 2]}
        data = Data(data=dataset, task=Task.CLASSIFICATION, label_name="label")
        data_copy = Data(data=dataset, task=Task.CLASSIFICATION, label_name="label")
        assert data != data_copy
        assert data != dataset

    def test_load_dataset_from_string(self, monkeypatch):
        # Test with non-existent file
        with pytest.raises(DatasetLoadError):
            Data._load_dataset_from_string("non_existent_file.txt")

        # Test with invalid extension
        with tempfile.NamedTemporaryFile(suffix=".invalid") as temp_file:
            with pytest.raises(DatasetLoadError):
                Data._load_dataset_from_string(temp_file.name)

        # Test with invalid external dataset identifier
        with patch("datasets.load_dataset") as mock_load_dataset:
            mock_load_dataset.side_effect = ValueError("Invalid external dataset identifier")
            with pytest.raises(DatasetLoadError) as excinfo:
                Data._load_dataset_from_string("invalid_external_dataset_name")

            expected_error_substring = "Failed to load dataset from <class 'str'>.\n"
            assert expected_error_substring in str(excinfo.value)

        # Test with valid .txt, .csv, and .json files
        test_data = [
            (".txt", "sample text", "from_text"),
            (".csv", "column1,column2\nvalue1,value2", "from_csv"),
            (".json", '{"key": "value"}', "from_json"),
        ]

        mock_dataset = Dataset.from_dict({"y": [1, 2, 3]})
        for ext, content, loader_func in test_data:
            with tempfile.NamedTemporaryFile(suffix=ext, mode="w+t") as temp_file:
                temp_file.write(content)
                temp_file.flush()

                # Make sure the correct loader function is called
                def fake_loader(file_name):
                    assert file_name == temp_file.name
                    return mock_dataset

                with monkeypatch.context() as mp:
                    mp.setattr(Dataset, loader_func, fake_loader)
                    loaded_dataset = Data._load_dataset_from_string(temp_file.name)
                    assert isinstance(loaded_dataset, Dataset)
                    assert loaded_dataset == mock_dataset

        # Test with an external dataset
        def fake_load_dataset(data_string):
            if data_string == "external_dataset":
                return mock_dataset

            raise Exception("Not the expected dataset string")

        with monkeypatch.context() as mp:
            mp.setattr("datasets.load_dataset", fake_load_dataset)
            loaded_dataset = Data._load_dataset_from_string("external_dataset")
            assert isinstance(loaded_dataset, Dataset)
            assert loaded_dataset == mock_dataset

            with pytest.raises(DatasetLoadError) as excinfo:
                Data._load_dataset_from_string("non_external_dataset")

            expected_error_substring = "Failed to load dataset from <class 'str'>.\n"

    def test_check_label_nan_with_dict_and_dataset(self):
        valid_data = {"features": [1, 2, 3], "labels": [0, 1, 0]}
        data_instance = Data(data=valid_data, task=Task.CLASSIFICATION, label_name="labels")
        data_instance._check_label_nan(valid_data, "labels")

        invalid_data = {"features": [1, 2, 3], "labels": [0, np.nan, 0]}
        with pytest.raises(LabelNanError) as exc_info:
            data_instance._check_label_nan(invalid_data, "labels")
        assert (
            "Found 1 NaN value(s) in the label column 'labels'. Please handle NaN values in before creating Datalab instance."
            in str(exc_info.value)
        )

        multiple_nan_data = {"features": [1, 2, 3, 4], "labels": [np.nan, 1, np.nan, 0]}
        with pytest.raises(LabelNanError) as exc_info:
            data_instance._check_label_nan(multiple_nan_data, "labels")
        assert (
            "Found 2 NaN value(s) in the label column 'labels'. Please handle NaN values in before creating Datalab instance."
            in str(exc_info.value)
        )

        valid_dataset = Dataset.from_dict(valid_data)

        data_instance = Data(data=valid_dataset, task=Task.CLASSIFICATION, label_name="labels")
        data_instance._check_label_nan(valid_dataset, "labels")

        invalid_dataset = Dataset.from_dict(invalid_data)
        with pytest.raises(LabelNanError) as exc_info:
            data_instance._check_label_nan(invalid_dataset, "labels")
        assert (
            "Found 1 NaN value(s) in the label column 'labels'. Please handle NaN values in before creating Datalab instance."
            in str(exc_info.value)
        )

    @given(dataset=dataset_strategy(task=Task.CLASSIFICATION))
    def test_label_map_is_lexicographically_ordered(self, dataset):
        data = Data(data=dataset, task=Task.CLASSIFICATION, label_name="label")
        label_map = data.labels.label_map
        assert list(label_map.values()) == sorted(label_map.values())

    @given(dataset=dataset_strategy(task=Task.MULTILABEL))
    def test_label_map_is_lexicographically_ordered_multilabel(self, dataset):
        data = Data(data=dataset, task=Task.MULTILABEL, label_name="label")
        label_map = data.labels.label_map
        assert list(label_map.values()) == sorted(label_map.values())
