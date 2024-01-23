<p align="center">
  <img src="https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanlab_logo_open_source_transparent_optimized_size.png" width=60% height=60%>
</p>


cleanlab helps you **clean** data and **lab**els by automatically detecting issues in a ML dataset. To facilitate **machine learning with messy, real-world data**, this data-centric AI package uses your *existing* models to estimate dataset problems that can be fixed to train even *better* models.

```python

# cleanlab works with **any classifier**. Yup, you can use PyTorch/TensorFlow/OpenAI/XGBoost/etc.
cl = cleanlab.classification.CleanLearning(sklearn.YourFavoriteClassifier())

# cleanlab finds data and label issues in **any dataset**... in ONE line of code!
label_issues = cl.find_label_issues(data, labels)

# cleanlab trains a robust version of your model that works more reliably with noisy data.
cl.fit(data, labels)

# cleanlab estimates the predictions you would have gotten if you had trained with *no* label issues.
cl.predict(test_data)

# A universal data-centric AI tool, cleanlab quantifies class-level issues and overall data quality, for any dataset.
cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint)
```

Get started with: [tutorials](https://docs.cleanlab.ai/stable/tutorials/image.html), [documentation](https://docs.cleanlab.ai/), [examples](https://github.com/cleanlab/examples), and [blogs](https://cleanlab.ai/blog/).

 - Learn to run cleanlab on your data in 5 minutes for: [image](https://docs.cleanlab.ai/stable/tutorials/image.html), [text](https://docs.cleanlab.ai/stable/tutorials/datalab/text.html), [audio](https://docs.cleanlab.ai/stable/tutorials/audio.html), or [tabular](https://docs.cleanlab.ai/stable/tutorials/datalab/tabular.html) data.
- Use cleanlab to automatically: [detect data issues (outliers, duplicates, label errors, etc)](https://docs.cleanlab.ai/stable/tutorials/datalab/datalab_quickstart.html), [train robust models](https://docs.cleanlab.ai/stable/tutorials/indepth_overview.html), [infer consensus + annotator-quality for multi-annotator data](https://docs.cleanlab.ai/stable/tutorials/multiannotator.html), [suggest data to (re)label next (active learning)](https://github.com/cleanlab/examples/blob/master/active_learning_multiannotator/active_learning.ipynb). 


[![pypi](https://img.shields.io/pypi/v/cleanlab.svg)](https://pypi.org/pypi/cleanlab/)
[![os](https://img.shields.io/badge/platform-noarch-lightgrey)](https://pypi.org/pypi/cleanlab/)
[![py\_versions](https://img.shields.io/badge/python-3.8%2B-blue)](https://pypi.org/pypi/cleanlab/)
[![build\_status](https://github.com/cleanlab/cleanlab/workflows/CI/badge.svg)](https://github.com/cleanlab/cleanlab/actions?query=workflow%3ACI)
[![coverage](https://codecov.io/gh/cleanlab/cleanlab/branch/master/graph/badge.svg)](https://app.codecov.io/gh/cleanlab/cleanlab)
[![docs](https://img.shields.io/static/v1?logo=github&style=flat&color=pink&label=docs&message=cleanlab)](https://docs.cleanlab.ai/)
[![Slack Community](https://img.shields.io/static/v1?logo=slack&style=flat&color=white&label=slack&message=community)](https://cleanlab.ai/slack)
[![Twitter](https://img.shields.io/twitter/follow/CleanlabAI?style=social)](https://twitter.com/CleanlabAI)
[![Cleanlab Studio](https://raw.githubusercontent.com/cleanlab/assets/master/shields/cl-studio-shield.svg)](https://cleanlab.ai/studio/?utm_source=github&utm_medium=readme&utm_campaign=clostostudio)


<p align="center">
  <img src="https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/datalab_issues.png" width=74% height=74%>
</p>
<p align="center">
    Examples of various issues in Cat/Dog dataset <b>automatically detected</b> by cleanlab via this code:    
</p>

```python
        lab = cleanlab.Datalab(data=dataset, label="column_name_for_labels")
        # Fit any ML model, get its feature_embeddings & pred_probs for your data
        lab.find_issues(features=feature_embeddings, pred_probs=pred_probs)
        lab.report()
```

## So fresh, so cleanlab

cleanlab **clean**s your data's **lab**els via state-of-the-art *confident learning* algorithms, published in this [paper](https://jair.org/index.php/jair/article/view/12125) and [blog](https://l7.curtisnorthcutt.com/confident-learning). See some of the datasets cleaned with cleanlab at [labelerrors.com](https://labelerrors.com). This data-centric AI tool helps you find data and label issues, so you can train reliable ML models.

cleanlab is:

1. **backed by theory** -- with [provable guarantees](https://arxiv.org/abs/1911.00068) of exact label noise estimation, even with imperfect models.
2. **fast** -- code is parallelized and scalable.
4. **easy to use** -- one line of code to find mislabeled data, bad annotators, outliers, or train noise-robust models.
6. **general** -- works with **[any dataset](https://labelerrors.com/)** (text, image, tabular, audio,...) + **any model** (PyTorch, OpenAI, XGBoost,...)
<br/>

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/label-errors-examples.png)
<p align="center">
Examples of incorrect given labels in various image datasets <a href="https://l7.curtisnorthcutt.com/label-errors">found and corrected</a> using cleanlab.
</p>

## Run cleanlab

cleanlab supports Linux, macOS, and Windows and runs on Python 3.8+.

- Get started [here](https://docs.cleanlab.ai/)! Install via `pip` or `conda` as described [here](https://docs.cleanlab.ai/).
- Developers who install the bleeding-edge from source should refer to [this master branch documentation](https://docs.cleanlab.ai/master/index.html).
- For help, check out our detailed [FAQ](https://docs.cleanlab.ai/stable/tutorials/faq.html), [Github Issues](https://github.com/cleanlab/cleanlab/issues?q=is%3Aissue), or [Slack](https://cleanlab.ai/slack). We welcome any questions!

**Practicing data-centric AI can look like this:**
1. Train initial ML model on original dataset.
2. Utilize this model to diagnose data issues (via cleanlab methods) and improve the dataset.
3. Train the same model on the improved dataset. 
4. Try various modeling techniques to further improve performance.

Most folks jump from Step 1 → 4, but you may achieve big gains without *any* change to your modeling code by using cleanlab!
Continuously boost performance by iterating Steps 2 → 4 (and try to evaluate with *cleaned* data).

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/dcai_flowchart.png)


## Use cleanlab with any model for most ML tasks

All features of cleanlab work with **any dataset** and **any model**. Yes, any model: PyTorch, Tensorflow, Keras, JAX, HuggingFace, OpenAI, XGBoost, scikit-learn, etc.
If you use a sklearn-compatible classifier, all cleanlab methods work out-of-the-box.

<details><summary>
It’s also easy to use your favorite non-sklearn-compatible model (<b>click to learn more</b>)
</summary>
<br/>

cleanlab can find label issues from any model's predicted class probabilities if you can produce them yourself.

Some cleanlab functionality may require your model to be sklearn-compatible.
There's nothing you need to do if your model already has `.fit()`, `.predict()`, and `.predict_proba()` methods.
Otherwise, just wrap your custom model into a Python class that inherits the `sklearn.base.BaseEstimator`:

``` python
from sklearn.base import BaseEstimator
class YourFavoriteModel(BaseEstimator): # Inherits sklearn base classifier
    def __init__(self, ):
        pass  # ensure this re-initializes parameters for neural net models
    def fit(self, X, y, sample_weight=None):
        pass
    def predict(self, X):
        pass
    def predict_proba(self, X):
        pass
    def score(self, X, y, sample_weight=None):
        pass
```

This inheritance allows to apply a wide range of sklearn functionality like hyperparameter-optimization to your custom model.
Now you can use your model with every method in cleanlab. Here's one example:

``` python
from cleanlab.classification import CleanLearning
cl = CleanLearning(clf=YourFavoriteModel())  # has all the same methods of YourFavoriteModel
cl.fit(train_data, train_labels_with_errors)
cl.predict(test_data)
```

#### Want to see a working example? [Here’s a compliant PyTorch MNIST CNN class](https://github.com/cleanlab/cleanlab/blob/master/cleanlab/experimental/mnist_pytorch.py)

More details are provided in documentation of [cleanlab.classification.CleanLearning](https://docs.cleanlab.ai/stable/cleanlab/classification.html).

Note, some libraries exist to give you sklearn-compatibility for free. For PyTorch, check out the [skorch](https://skorch.readthedocs.io/) Python library which will wrap your PyTorch model into a sklearn-compatible model ([example](https://docs.cleanlab.ai/stable/tutorials/image.html)). For TensorFlow/Keras, check out our [Keras wrapper](https://docs.cleanlab.ai/stable/cleanlab/models/keras.html). Many libraries also already offer a special scikit-learn API, for example: [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn) or [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html).

<br/>
</details>

cleanlab is useful across a wide variety of Machine Learning tasks. Specific tasks this data-centric AI solution offers dedicated functionality for include:
1. [Binary and multi-class classification](https://docs.cleanlab.ai/stable/tutorials/indepth_overview.html)
2. [Multi-label classification](https://docs.cleanlab.ai/stable/tutorials/multilabel_classification.html) (e.g. image/document tagging)
3. [Token classification](https://docs.cleanlab.ai/stable/tutorials/token_classification.html) (e.g. entity recognition in text)
4. [Regression](https://docs.cleanlab.ai/stable/tutorials/regression.html) (predicting numerical column in a dataset)
5. [Image segmentation](https://docs.cleanlab.ai/stable/tutorials/segmentation.html) (images with per-pixel annotations)
6. [Object detection](https://docs.cleanlab.ai/stable/tutorials/object_detection.html) (images with bounding box annotations)
7. [Classification with data labeled by multiple annotators](https://docs.cleanlab.ai/stable/tutorials/multiannotator.html)
8. [Active learning with multiple annotators](https://github.com/cleanlab/examples/blob/master/active_learning_multiannotator/active_learning.ipynb) (suggest which data to label or re-label to improve model most)
9. [Outlier detection](https://docs.cleanlab.ai/stable/tutorials/outliers.html) (identify atypical data that appears out of distribution)

For other ML tasks, cleanlab can still help you improve your dataset if appropriately applied.
Many practical applications are demonstrated in our [Example Notebooks](https://github.com/cleanlab/examples).


## Citation and related publications

cleanlab is based on peer-reviewed research. Here are relevant papers to cite if you use this package:

<details><summary><a href="https://arxiv.org/abs/1911.00068">Confident Learning (JAIR '21)</a> (<b>click to show bibtex</b>) </summary>

    @article{northcutt2021confidentlearning,
        title={Confident Learning: Estimating Uncertainty in Dataset Labels},
        author={Curtis G. Northcutt and Lu Jiang and Isaac L. Chuang},
        journal={Journal of Artificial Intelligence Research (JAIR)},
        volume={70},
        pages={1373--1411},
        year={2021}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/1705.01936">Rank Pruning (UAI '17)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{northcutt2017rankpruning,
        author={Northcutt, Curtis G. and Wu, Tailin and Chuang, Isaac L.},
        title={Learning with Confident Examples: Rank Pruning for Robust Classification with Noisy Labels},
        booktitle = {Proceedings of the Thirty-Third Conference on Uncertainty in Artificial Intelligence},
        series = {UAI'17},
        year = {2017},
        location = {Sydney, Australia},
        numpages = {10},
        url = {http://auai.org/uai2017/proceedings/papers/35.pdf},
        publisher = {AUAI Press},
    }

</details>

<details><summary><a href="https://people.csail.mit.edu/jonasmueller/info/LabelQuality_icml.pdf"> Label Quality Scoring (ICML '22)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{kuan2022labelquality,
        title={Model-agnostic label quality scoring to detect real-world label errors},
        author={Kuan, Johnson and Mueller, Jonas},
        booktitle={ICML DataPerf Workshop},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2207.03061"> Out-of-Distribution Detection (ICML '22)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{kuan2022ood,
        title={Back to the Basics: Revisiting Out-of-Distribution Detection Baselines},
        author={Kuan, Johnson and Mueller, Jonas},
        booktitle={ICML Workshop on Principles of Distribution Shift},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2210.03920"> Token Classification Label Errors (NeurIPS '22)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{wang2022tokenerrors,
        title={Detecting label errors in token classification data},
        author={Wang, Wei-Chen and Mueller, Jonas},
        booktitle={NeurIPS Workshop on Interactive Learning for Natural Language Processing (InterNLP)},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2210.06812"> CROWDLAB for Data with Multiple Annotators (NeurIPS '22)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{goh2022crowdlab,
        title={CROWDLAB: Supervised learning to infer consensus labels and quality scores for data with multiple annotators},
        author={Goh, Hui Wen and Tkachenko, Ulyana and Mueller, Jonas},
        booktitle={NeurIPS Human in the Loop Learning Workshop},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2301.11856"> ActiveLab: Active learning with data re-labeling (ICLR '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{goh2023activelab,
        title={ActiveLab: Active Learning with Re-Labeling by Multiple Annotators},
        author={Goh, Hui Wen and Mueller, Jonas},
        booktitle={ICLR Workshop on Trustworthy ML},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2211.13895"> Incorrect Annotations in Multi-Label Classification (ICLR '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{thyagarajan2023multilabel,
        title={Identifying Incorrect Annotations in Multi-Label Classification Data},
        author={Thyagarajan, Aditya and Snorrason, Elías and Northcutt, Curtis and Mueller, Jonas},
        booktitle={ICLR Workshop on Trustworthy ML},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2305.15696"> Detecting Dataset Drift and Non-IID Sampling (ICML '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{cummings2023drift,
        title={Detecting Dataset Drift and Non-IID Sampling via k-Nearest Neighbors},
        author={Cummings, Jesse and Snorrason, Elías and Mueller, Jonas},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2305.16583"> Detecting Errors in Numerical Data (ICML '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{zhou2023errors,
        title={Detecting Errors in Numerical Data via any Regression Model},
        author={Zhou, Hang and Mueller, Jonas and Kumar, Mayank and Wang, Jane-Ling and Lei, Jing},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2309.00832"> ObjectLab: Mislabeled Images in Object Detection Data (ICML '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{tkachenko2023objectlab,
        title={ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data},
        author={Tkachenko, Ulyana and Thyagarajan, Aditya and Mueller, Jonas},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2307.05080"> Label Errors in Segmentation Data (ICML '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{lad2023segmentation,
        title={Estimating label quality and errors in semantic segmentation data via any model},
        author={Lad, Vedang and Mueller, Jonas},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

To understand/cite other cleanlab functionality not described above, check out our [additional publications](https://cleanlab.ai/research/).


## Other resources

- [Example Notebooks demonstrating practical applications of this package](https://github.com/cleanlab/examples)

- [Cleanlab Blog](https://cleanlab.ai/blog/)

- [Blog post: Introduction to Confident Learning](https://l7.curtisnorthcutt.com/confident-learning)

- [NeurIPS 2021 paper: Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks](https://arxiv.org/abs/2103.14749)

- [Introduction to Data-centric AI (MIT IAP Course 2023)](https://dcai.csail.mit.edu/)

- [Release notes for past versions](https://github.com/cleanlab/cleanlab/releases)

- [Cleanlab Studio](https://cleanlab.ai/studio/?utm_source=github&utm_medium=readme&utm_campaign=clostostudio): *No-code Data Improvement*

While this open-source library **finds** data issues, its utility depends on you having a decent existing ML model and an interface to efficiently **fix** these issues in your dataset. Providing all these pieces, [Cleanlab Studio](https://cleanlab.ai/studio/?utm_source=github&utm_medium=readme&utm_campaign=clostostudio) is a no-code platform to **find and fix** problems in real-world ML datasets. Cleanlab Studio [automatically runs](https://cleanlab.ai/blog/data-centric-ai/) optimized versions of the algorithms from this open-source library on top of AutoML & Foundation models fit to your data, and presents detected issues in a smart data editing interface. It's a data cleaning assistant to quickly turn unreliable data into reliable models/insights (via AI/automation + streamlined UX). [Try it for free!](https://cleanlab.ai/signup)

<p align="center">
  <img src="https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/studio.png" width=80% height=80% alt="Cleanlab Studio logo">
</p>

## Join our community

* The best place to learn is [our Slack community](https://cleanlab.ai/slack).

* Have ideas for the future of cleanlab? How are you using cleanlab? [Join the discussion](https://github.com/cleanlab/cleanlab/discussions) and check out [our active/planned Projects and what we could use your help with](https://github.com/cleanlab/cleanlab/projects).

* Interested in contributing? See the [contributing guide](CONTRIBUTING.md) and [ideas on useful contributions](https://github.com/cleanlab/cleanlab/wiki#ideas-for-contributing-to-cleanlab). We welcome your help building a standard open-source platform for data-centric AI!

* Have code improvements for cleanlab? See the [development guide](DEVELOPMENT.md).

* Have an issue with cleanlab? Search [our FAQ](https://docs.cleanlab.ai/stable/tutorials/faq.html) and [existing issues](https://github.com/cleanlab/cleanlab/issues?q=is%3Aissue), or [submit a new issue](https://github.com/cleanlab/cleanlab/issues/new).

* Need professional help with cleanlab?
Join our [\#help Slack channel](https://cleanlab.ai/slack) and message us there, or reach out via email: team@cleanlab.ai

## License

Copyright (c) 2017 Cleanlab Inc.

cleanlab is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

cleanlab is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

See [GNU Affero General Public LICENSE](https://github.com/cleanlab/cleanlab/blob/master/LICENSE) for details.
You can email us to discuss licensing: team@cleanlab.ai

### Commercial licensing

Commercial licensing is available for teams and enterprises that want to use cleanlab in production workflows, but are unable to open-source their code [as is required by the current license](https://github.com/cleanlab/cleanlab/blob/master/LICENSE). Please email us: team@cleanlab.ai
