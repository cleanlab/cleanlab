import sys
import importlib
from unittest.mock import patch


def test_datalab_unavailable():
    with patch.dict(sys.modules, {"cleanlab.datalab.datalab": ImportError("Mocked ImportError")}):
        # Reload the module to trigger the import statement
        import cleanlab

        importlib.reload(cleanlab)

        assert cleanlab.Datalab.message == (
            "Datalab is not available due to missing dependencies. "
            "To install Datalab, run `pip install 'cleanlab[datalab]'`."
        )
