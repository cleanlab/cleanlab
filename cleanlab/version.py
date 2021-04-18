# Copyright (c) 2017-2050 Curtis G. Northcutt

# This file is part of cleanlab.
#
# cleanlab is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cleanlab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License

# This agreement applies to this version and all previous versions of cleanlab.


__version__ = '1.0'


# ----------------------------------------
# | CURRENT STABLE VERSION RELEASE NOTES |
# ----------------------------------------

# 1.0 - cleanlab official 1.0 (beta) release!
#   - Added Amazon Reviews NLP to cleanlab/examples
#   - cleanlab now supports python 2, 2.7, 3.4, 3.5, 3.6, 3.7, 3.8.
#   - Users have used cleanlab with python version 3.9 (use at your own risk!)
#   - Added more testing. All tests pass on windows/linux/macOS.
#   - Update to GNU GPL-3+ License.
#   - Added documentation: https://cleanlab.readthedocs.io/
#   - The cleanlab "confident learning" paper is published in the Journal of AI Research: https://jair.org/index.php/jair/article/view/12125
#   - Added funding, community and contributing guidelines
#   - Fixed a number of errors in cleanlab/examples
#   - cleanlab now supports Windows, macOS, Linux, and unix systems
#   - Numerous examples added to the README and docs
#   - cleanlab now natively supports Co-Teaching for learning with noisy labels (reqs python3, PyTorch 1.4)
#   - cleanlab built in support with handwritten datasets (besides MNIST)
#   - cleanlab built in support for CIFAR dataset
#   - Multiprocessing fixed for windows systems
#   - Adhered all core modules to PEP-8 styling.
#   - Extensive benchmarking of cleanlab methods published.
#   - Future features planned are now supported in cleanlab/version.py
#   - Added confidentlearning-reproduce as a seperate repo to reproduce state-of-the-art results.

# ---------------------------
# | FUTURE FEATURES PLANNED |
# ---------------------------

#   - Extensions to regression
#   - Extensions for object detection and segmentation tasks.
#   - cleanlab pro (interface for businesses)

# ----------------------------------
# | PREVIOUS VERSION RELEASE NOTES |
# ----------------------------------

# 0.1.1 - Major update adding support for Windows and Python 3.7
#   - Added support for Python 3.7
#   - Added full support for Windows, including multiprocessing parallelization support in cleanlab/pruning.py
#   - Improved PEP-8 adherence in core cleanlab/ code.

# 0.1.0 - Release of confident learning paper based on this version of cleanlab: https://arxiv.org/pdf/1911.00068.pdf
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
#   - estimate_confident_joint_from_probabilities now automatically calibrates confident joint to be a true joint estimate.
#   - Confident joint algorithm changed! When an example is found confidently as 2+ labels, choose class with max probability.

# 0.0.7 - Massive speed increases across the board. Estimating confident joint now nearly instant. NO major API changes.

# 0.0.6 - NO API changes. README updates. Examples added. Tutorials added.

# 0.0.5 - Numerous small bug fixes, but not major API changes. 100% testing code coverage.

# 0.0.4 - FIRST CROSS-PLATFORM WORKING VERSION OF CLEANLAB. Adding test support.

# 0.0.3 - Adding working logo to README, pypi working

# 0.0.2 - Added logo to README, but link does not load on pypi

# 0.0.1 - initial commit
