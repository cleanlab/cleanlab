<div align="center">
  <img src="https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/cleanlab_logo_open_source.png" width=60%>
</div>

<div align="center">
<a href="https://pypi.org/pypi/cleanlab/" target="_blank"><img src="https://img.shields.io/pypi/v/cleanlab.svg" alt="pypi_versions"></a>
<a href="https://pypi.org/pypi/cleanlab/" target="_blank"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="py_versions"></a>
<a href="https://app.codecov.io/gh/cleanlab/cleanlab" target="_blank"><img src="https://codecov.io/gh/cleanlab/cleanlab/branch/master/graph/badge.svg" alt="coverage"></a>
<a href="https://github.com/cleanlab/cleanlab/stargazers/" target="_blank"><img src="https://img.shields.io/github/stars/cleanlab/cleanlab?style=social&maxAge=2592000" alt="Github Stars"></a>
<a href="https://twitter.com/CleanlabAI" target="_blank"><img src="https://img.shields.io/twitter/follow/CleanlabAI?style=social" alt="Twitter"></a>
</div>

<h4 align="center">
    <p>
        <a href="https://docs.cleanlab.ai/">Documentation</a> |
        <a href="https://github.com/cleanlab/examples">Examples</a> |
        <a href="https://cleanlab.ai/blog/learn/">Blog</a> |
        <a href="#citation-and-related-publications">Research</a>
    <p>
</h4>

Cleanlab’s open-source library helps you **clean** data and **lab**els by automatically detecting issues in a ML dataset. To facilitate **machine learning with messy, real-world data**, this data-centric AI package uses your *existing* models to estimate dataset problems that can be fixed to train even *better* models.
 

<p align="center">
  <img src="https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/datalab_issues.png" width=74%>
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

- Use cleanlab to automatically check every: [text](https://docs.cleanlab.ai/stable/tutorials/datalab/text.html), [audio](https://docs.cleanlab.ai/stable/tutorials/datalab/audio.html), [image](https://docs.cleanlab.ai/stable/tutorials/datalab/image.html), or [tabular](https://docs.cleanlab.ai/stable/tutorials/datalab/tabular.html) dataset.
- Use cleanlab to automatically: [detect data issues (outliers, duplicates, label errors, etc)](https://docs.cleanlab.ai/stable/tutorials/datalab/datalab_quickstart.html), [train robust models](https://docs.cleanlab.ai/stable/tutorials/indepth_overview.html), [infer consensus + annotator-quality for multi-annotator data](https://docs.cleanlab.ai/stable/tutorials/multiannotator.html), [suggest data to (re)label next (active learning)](https://github.com/cleanlab/examples/blob/master/active_learning_multiannotator/active_learning.ipynb).


---


## Run cleanlab open-source

This cleanlab package runs on Python 3.10+ and supports Linux, macOS, as well as Windows.

- Get started [here](https://docs.cleanlab.ai/)! Install via `pip` or `conda`.
- Developers who install the bleeding-edge from source should refer to [this master branch documentation](https://docs.cleanlab.ai/master/index.html).

**Practicing data-centric AI can look like this:**
1. Train initial ML model on original dataset.
2. Utilize this model to diagnose data issues (via cleanlab methods) and improve the dataset.
3. Train the same model on the improved dataset. 
4. Try various modeling techniques to further improve performance.

Most folks jump from Step 1 → 4, but you may achieve big gains without *any* change to your modeling code by using cleanlab!
Continuously boost performance by iterating Steps 2 → 4 (and try to evaluate with *cleaned* data).

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/flowchart.png)


## Use cleanlab with any model and in most ML tasks

All features of cleanlab work with **any dataset** and **any model**. Yes, any model: PyTorch, Tensorflow, Keras, JAX, HuggingFace, OpenAI, XGBoost, scikit-learn, etc.

cleanlab is useful across a wide variety of Machine Learning tasks. Specific tasks this data-centric AI package offers dedicated functionality for include:
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
See our [Example Notebooks](https://github.com/cleanlab/examples) and [Blog](https://cleanlab.ai/blog/learn/).


## So fresh, so cleanlab

Beyond automatically catching [all sorts of issues](https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html) lurking in your data, this data-centric AI package helps you deal with **noisy labels** and train more **robust ML models**.
Here's an example:

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

cleanlab **clean**s your data's **lab**els via state-of-the-art *confident learning* algorithms, published in this [paper](https://jair.org/index.php/jair/article/view/12125) and [blog](https://l7.curtisnorthcutt.com/confident-learning). See some of the datasets cleaned with cleanlab at [labelerrors.com](https://labelerrors.com).

cleanlab is:

1. **backed by theory** -- with [provable guarantees](https://arxiv.org/abs/1911.00068) of exact label noise estimation, even with imperfect models.
2. **fast** -- code is parallelized and scalable.
4. **easy to use** -- one line of code to find mislabeled data, bad annotators, outliers, or train noise-robust models.
6. **general** -- works with **[any dataset](https://labelerrors.com/)** (text, image, tabular, audio,...) + **any model** (PyTorch, OpenAI, XGBoost,...)
<br/>

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/label-errors-examples.png)
<p align="center">
Examples of incorrect given labels in various image datasets <a href="https://l7.curtisnorthcutt.com/label-errors">found and corrected</a> using cleanlab. 
While these examples are from image datasets, this also works for text, audio, tabular data.
</p>


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

<details><summary><a href="https://jonasmueller.org/info/LabelQuality_icml.pdf"> Label Quality Scoring (ICML '22)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{kuan2022labelquality,
        title={Model-agnostic label quality scoring to detect real-world label errors},
        author={Kuan, Johnson and Mueller, Jonas},
        booktitle={ICML DataPerf Workshop},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2210.03920"> Label Errors in Token Classification / Entity Recognition (NeurIPS '22)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{wang2022tokenerrors,
        title={Detecting label errors in token classification data},
        author={Wang, Wei-Chen and Mueller, Jonas},
        booktitle={NeurIPS Workshop on Interactive Learning for Natural Language Processing (InterNLP)},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2211.13895"> Label Errors in Multi-Label Classification (ICLR '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{thyagarajan2023multilabel,
        title={Identifying Incorrect Annotations in Multi-Label Classification Data},
        author={Thyagarajan, Aditya and Snorrason, Elías and Northcutt, Curtis and Mueller, Jonas},
        booktitle={ICLR Workshop on Trustworthy ML},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2309.00832"> Label Errors in Object Detection (ICML '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{tkachenko2023objectlab,
        title={ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data},
        author={Tkachenko, Ulyana and Thyagarajan, Aditya and Mueller, Jonas},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2307.05080"> Label Errors in Image Segmentation (ICML '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{lad2023segmentation,
        title={Estimating label quality and errors in semantic segmentation data via any model},
        author={Lad, Vedang and Mueller, Jonas},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2305.16583"> Detecting Errors in Numerical Data (DMLR '24)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{zhou2023errors,
        title={Detecting Errors in a Numerical Response via any Regression Model},
        author={Zhou, Hang and Mueller, Jonas and Kumar, Mayank and Wang, Jane-Ling and Lei, Jing},
        booktitle={Journal of Data-centric Machine Learning Research},
        year={2024}
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

<details><summary><a href="https://arxiv.org/abs/2305.15696"> Detecting Dataset Drift and Non-IID Sampling (ICML '23)</a> (<b>click to show bibtex</b>) </summary>

    @inproceedings{cummings2023drift,
        title={Detecting Dataset Drift and Non-IID Sampling via k-Nearest Neighbors},
        author={Cummings, Jesse and Snorrason, Elías and Mueller, Jonas},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

To understand/cite other cleanlab functionality not described above, check out our [Blog](https://cleanlab.ai/blog/learn).


## Other resources

- [Example Notebooks demonstrating practical applications of this package](https://github.com/cleanlab/examples)

- [Cleanlab Blog](https://cleanlab.ai/blog/learn)

- [Blog post: Introduction to Confident Learning](https://l7.curtisnorthcutt.com/confident-learning)

- [NeurIPS 2021 paper: Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks](https://arxiv.org/abs/2103.14749)

- [Introduction to Data-centric AI (MIT IAP Course)](https://dcai.csail.mit.edu/)

- [Release notes for past versions](https://github.com/cleanlab/cleanlab/releases)

**Interested in contributing?**  See the [contributing guide](CONTRIBUTING.md), [development guide](DEVELOPMENT.md), and [ideas on useful contributions](https://github.com/cleanlab/cleanlab/wiki#ideas-for-contributing-to-cleanlab).

**Have questions?**  Check out [our FAQ](https://docs.cleanlab.ai/stable/tutorials/faq.html) and [Github Issues](https://github.com/cleanlab/cleanlab/issues?q=is%3Aissue).
