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


__version__ = "2.6.0"

# ------------------------------------------------
# | PREVIOUS MAJOR VERSION RELEASE NOTES SUMMARY |
# ------------------------------------------------

# 2.6.0 - Elevating Data Insights: Comprehensive Issue Checks & Expanded ML Task Compatibility
#
#   Major new functionalities include:
#   - Detection for null values, class imbalance, underperforming groups and data valuation in Datalab
#   - Extend Datalab support for tasks like regression and multi-label classification
#   - Scores for near duplicates and outliers rescaled to be more interpretable

# 2.5.0 - cleanlab detects label errors in most ML tasks
#
#   Major new functionalities include:
#   - Support for: regression, object detection, image segmentation
#   - Detection of low-quality images

# 2.4.0 - One line of code to detect all sorts of dataset issues
#
#   Major new functionalities include:
#   - Datalab: A unified audit to detect different types of issues in your data and labels. This is the primary way most users should apply cleanlab to their dataset.
#   - Nicer APIs for label issues in multi-label classification datasets, including dataset-level issue summaries for multi-label classification.
#   - Updated tutorials with more interesting datasets and ML models.

# 2.3.0 - Extending cleanlab beyond label errors into a complete library for data-centric AI
#
#   Major new functionalities include:
#   - Active learning with data re-labeling (ActiveLab)
#   - KerasWrapperModel and KerasSequentialWrapper to make arbitrary Keras models compatible with scikit-learn
#   - Computational improvements for detecting label issues (better efficiency and mini-batch estimation that works with lower memory)

# 2.2.0 - Re-invented algorithms for multi-label classification and support for datasets with missing classes
#
#   For detecting label errors in multi-label classification datasets (e.g. image/document tagging):
#   - Added cleanlab.multilabel_classification module for label quality scoring.
#   - Re-invented better algorithms and published paper describing how the new algorithms work and benchmarking their effectiveness.
#   - Provided new quickstart tutorials and examples on how to use cleanlab for multi-label datasets and train effective models for them.
#
#   Additional improvements:
#   - cleanlab now works much better for datasets in which some classes happen to not be present.
#   - Algorithmic improvements to ensure count.num_label_issues() returns more accurate estimates.
#   - For developers: introduction of flake8 code linter and more comprehensive mypy type annotations.

# 2.1.0 - "Multiannotator, Outlier detection, and Token Classification" - Cleanlab supports several new features
#
#   For users (+ sometimes developers):
#   - Improved CleanLearning. Added support for pd.DataFrame, tf.keras.dataset, and other types of data besides np.ndarray
#   - Added cleanlab.multiannotator module for working with data labeled by multiple annotators.
#   - Added cleanlab.token_classification for token classification with text data.
#   - Added cleanlab.outlier for out-of-distribution detection (includes outlier/anomaly detection)
#
#   For developers:
#   - No significant API-breaking changers.
#   - Tutorials for all new modules added to https://docs.cleanlab.ai
#   - Contributing resources added to https://github.com/cleanlab/cleanlab/blob/master/CONTRIBUTING.md
#   - Contributing ideas added to https://github.com/cleanlab/cleanlab/wiki#ideas-for-contributing-to-cleanlab

# 2.0.0 - "Data-centric AI Ready" - Complete re-architecture of cleanlab API.
#
#   For users (+ sometimes developers):
#   - All aspects of API have changed (method names, parameters, defaults, variables, classes, etc)
#   - Added new dataset module for dealing with dataset-level issues
#   - CleanLearning now handles most cleanlab tasks in one line of code.
#   - Several new workflows possible with rank, count, and filter modules
#
#   For developers:
#   - If you're coming from 1.0 (pre-1.0.1), you may need to re-clone.
#   - Extensive support available at https://docs.cleanlab.ai

# 1.0.1 - Launch sphinx docs for Cleanlab 1.0 (in preparation for Cleanlab 2.0). Mostly superficial.
#
#   For users (+ sometimes developers):
#   - This releases the new sphinx docs for cleanlab 1.0 documentation (in preparation for CL 2.0)
#   - Several superficial bug fixes (reduce error printing, fix broken urls, clarify links)
#   - Extensive docs/README updates
#   - Support was added for Conda Installation
#   - Moved to AGPL-3 license
#   - Added tutorials and a learning section for Cleanlab
#
#   For developers:
#   - Moved to GitHub Actions CI
#   - Significantly shrunk the clone size to a few MB from 100MB+

# 1.0 - cleanlab official 1.0 (beta) release!
#   - Added Amazon Reviews NLP to cleanlab/examples
#   - cleanlab now supports python 2, 2.7, 3.4, 3.5, 3.6, 3.7, 3.8.
#   - Users have used cleanlab with python version 3.9 (use at your own risk!)
#   - Added more testing. All tests pass on windows/linux/macOS.
#   - Update to GNU GPL-3+ License.
#   - Added documentation: https://cleanlab.readthedocs.io/
#   - The cleanlab "confident learning" paper is published in the Journal of AI Research:
#       https://jair.org/index.php/jair/article/view/12125
#   - Added funding, community and contributing guidelines
#   - Fixed a number of errors in cleanlab/examples
#   - cleanlab now supports Windows, macOS, Linux, and unix systems
#   - Numerous examples added to the README and docs
#   - now natively supports Co-Teaching for learning with noisy labels, req: py3, PyTorch 1.4
#   - cleanlab built in support with handwritten datasets (besides MNIST)
#   - cleanlab built in support for CIFAR dataset
#   - Multiprocessing fixed for windows systems
#   - Adhered all core modules to PEP-8 styling
#   - Extensive benchmarking of cleanlab methods published.
#   - Future features planned are now supported in cleanlab/version.py
#   - Added confidentlearning-reproduce as a separate repo to reproduce state-of-the-art results.

# 0.1.1 - Major update adding support for Windows and Python 3.7
#   - Added support for Python 3.7
#   - Added full support for Windows, including multiprocessing support in cleanlab/filter.py
#   - Improved PEP-8 adherence in core cleanlab/ code.

# 0.1.0 - Release of confident learning paper: https://arxiv.org/pdf/1911.00068.pdf
#   - Documentation increase
#   - Add examples to find label errors in mnist, cifar, imagenet
#   - re-organized examples and added readme.

# 0.0.14 - Major bug fix in classification. Unused param broke code.

# 0.0.13 - Major bug fix in finding label errors.
#   - Fixed an important bug that broke finding label errors correctly.
#   - Added baseline methods for finding label errors and estimating joint
#   - Increased testing
#   - Simplified logic

# 0.0.12 - Minor changes.
#   - Added support and testing for sparse matrices scipy.sparse.csr_matrix
#   - Dropped integrated dependency and support on fasttext. Use fasttext at your own risk.
#   - Added testing and dropping fasttext bumped testing code coverage up to 96%.
#   - Remove all ipynb artifacts of the form # In [ ].

# 0.0.11 - New logo! Improved README.

# 0.0.10 - Improved documentation, code formatting, README, and testing coverage.

# 0.0.9 - Multiple major changes
#   - Important: refactored all confident joint methods and parameters
#   - Numerous important bug fixes
#   - Added multi_label support for labels (list of lists)
#   - Added automated ordering of label errors
#   - Added automatic calibration of the confident joint
#   - Version 0.0.8 is deprecated. Use this version going forward.

# 0.0.8 - Multiple major changes
#   - Finding label errors is now fully parallelized.
#   - prune_count_method parameter has been removed.
#   - estimate_confident_joint_from_probabilities now automatically calibrates confident joint
#       to be a true joint estimate.
#   - Confident joint algorithm changed! When an example is found confidently as 2+ labels, choose
#       class with max probability.

# 0.0.7 - Massive speed increases across the board. Estimate joint nearly instantly. NO API changes.

# 0.0.6 - NO API changes. README updates. Examples added. Tutorials added.

# 0.0.5 - Numerous small bug fixes, but not major API changes. 100% testing code coverage.

# 0.0.4 - FIRST CROSS-PLATFORM WORKING VERSION OF CLEANLAB. Adding test support.

# 0.0.3 - Adding working logo to README, pypi working

# 0.0.2 - Added logo to README, but link does not load on pypi

# 0.0.1 - initial commit
