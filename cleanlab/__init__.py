from .version import __version__
from . import classification
from . import count
from . import rank
from . import filter
from . import benchmarking
from . import dataset
from . import multiannotator
from . import outlier
from . import token_classification
from . import multilabel_classification
from . import object_detection
from . import regression
from . import segmentation


class DatalabUnavailable:
    def __init__(self, message):
        self.message = message

    def __getattr__(self, name):
        message = self.message + f" (raised when trying to access {name})"
        raise ImportError(message)

    def __call__(self, *args, **kwargs):
        message = (
            self.message + f" (raised when trying to call with args: {args}, kwargs: {kwargs})"
        )
        raise ImportError(message)


def _datalab_import_factory():
    try:
        from .datalab.datalab import Datalab as _Datalab

        return _Datalab
    except ImportError:
        return DatalabUnavailable(
            "Datalab is not available due to missing dependencies. "
            "To install Datalab, run `pip install 'cleanlab[datalab]'`."
        )


def _issue_manager_import_factory():
    try:
        from .datalab.internal.issue_manager import IssueManager as _IssueManager

        return _IssueManager
    except ImportError:
        return DatalabUnavailable(
            "IssueManager is not available due to missing dependencies for Datalab. "
            "To install Datalab, run `pip install 'cleanlab[datalab]'`."
        )


Datalab = _datalab_import_factory()
IssueManager = _issue_manager_import_factory()
