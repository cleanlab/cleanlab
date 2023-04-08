import pytest
from datasets.arrow_dataset import Dataset

from cleanlab.datalab.datalab import Datalab
import numpy as np

SEED = 42
LABEL_NAME = "star"


@pytest.fixture
def dataset():
    data_dict = {
        "id": [
            "7bd227d9-afc9-11e6-aba1-c4b301cdf627",
            "7bd22905-afc9-11e6-a5dc-c4b301cdf627",
            "7bd2299c-afc9-11e6-85d6-c4b301cdf627",
            "7bd22a26-afc9-11e6-9309-c4b301cdf627",
            "7bd22aba-afc9-11e6-8293-c4b301cdf627",
        ],
        "package_name": [
            "com.mantz_it.rfanalyzer",
            "com.mantz_it.rfanalyzer",
            "com.mantz_it.rfanalyzer",
            "com.mantz_it.rfanalyzer",
            "com.mantz_it.rfanalyzer",
        ],
        "review": [
            "Great app! The new version now works on my Bravia Android TV which is great as it's right by my rooftop aerial cable. The scan feature would be useful...any ETA on when this will be available? Also the option to import a list of bookmarks e.g. from a simple properties file would be useful.",
            "Great It's not fully optimised and has some issues with crashing but still a nice app  especially considering the price and it's open source.",
            "Works on a Nexus 6p I'm still messing around with my hackrf but it works with my Nexus 6p  Trond usb-c to usb host adapter. Thanks!",
            "The bandwidth seemed to be limited to maximum 2 MHz or so. I tried to increase the bandwidth but not possible. I purchased this is because one of the pictures in the advertisement showed the 2.4GHz band with around 10MHz or more bandwidth. Is it not possible to increase the bandwidth? If not  it is just the same performance as other free APPs.",
            "Works well with my Hackrf Hopefully new updates will arrive for extra functions",
        ],
        "date": [
            "October 12 2016",
            "August 23 2016",
            "August 04 2016",
            "July 25 2016",
            "July 22 2016",
        ],
        "star": [4, 4, 5, 3, 5],
        "version_id": [1487, 1487, 1487, 1487, 1487],
    }
    return Dataset.from_dict(data_dict)


@pytest.fixture
def label_name():
    return LABEL_NAME


@pytest.fixture
def lab(dataset, label_name):
    return Datalab(data=dataset, label_name=label_name)


@pytest.fixture
def pred_probs(dataset):
    np.random.seed(SEED)
    return np.random.rand(len(dataset), 3)
