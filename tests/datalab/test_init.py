import sys
import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cleanlab import Datalab


def test_datalab_unavailable():
    with patch.dict(sys.modules, {"cleanlab.datalab.datalab": ImportError("Mocked ImportError")}):
        # Reload the module to trigger the import statement
        import cleanlab

        importlib.reload(cleanlab)

        assert cleanlab.Datalab.message == (
            "Datalab is not available due to missing dependencies. "
            "To install Datalab, run `pip install 'cleanlab[datalab]'`."
        )


@pytest.mark.parametrize("invalid_label", [np.nan, None])
def test_datalab_init_raises_with_null_labels(invalid_label):
    data = pd.DataFrame({"text": ["a", "b", "c"], "label": [0, invalid_label, 1]})

    with pytest.raises(ValueError, match="Label column 'label' contains null or NaN values"):
        Datalab(data=data, label_name="label")
