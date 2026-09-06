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

<p align="center">
    <a href="README.md">English</a> · <b>简体中文</b>
</p>

<h4 align="center">
    <p>
        <a href="https://docs.cleanlab.ai/">官方文档</a> |
        <a href="https://github.com/cleanlab/examples">示例合集</a> |
        <a href="https://cleanlab.ai/blog/learn/">技术博客</a> |
        <a href="#论文引用与相关科研成果">论文研究</a>
    </p>
</h4>

Cleanlab 开源库通过自动检测机器学习数据集中的各类问题，帮助你**清洗（clean）**数据与**标签（labels）**。为了解决“**面对真实世界杂乱数据的机器学习建模**”难题，这个以数据为中心（Data-Centric AI）的软件包直接借助你*现有的*模型来诊断和估计数据集缺陷，并在修复数据后训练出效果更出色的模型。
 

<p align="center">
  <img src="https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/datalab_issues.png" width=74%>
</p>
<p align="center">
    以下是在猫/狗图像数据集中通过以下简短代码被 cleanlab <b>自动检测</b>出的各类真实数据缺陷：
</p>

```python
        lab = cleanlab.Datalab(data=dataset, label="column_name_for_labels")
        # 拟合任意机器学习模型，获取数据的特征嵌入 (feature_embeddings) 与预测概率 (pred_probs)
        lab.find_issues(features=feature_embeddings, pred_probs=pred_probs)
        lab.report()
```

- 使用 cleanlab 自动检测每种数据模态：[文本 (Text)](https://docs.cleanlab.ai/stable/tutorials/datalab/text.html)、[音频 (Audio)](https://docs.cleanlab.ai/stable/tutorials/datalab/audio.html)、[图像 (Image)](https://docs.cleanlab.ai/stable/tutorials/datalab/image.html) 或 [表格 (Tabular)](https://docs.cleanlab.ai/stable/tutorials/datalab/tabular.html) 数据集。
- 使用 cleanlab 实现全自动化：[检测数据问题（离群值、近似重复、标签错误等）](https://docs.cleanlab.ai/stable/tutorials/datalab/datalab_quickstart.html)、[训练抗噪鲁棒模型](https://docs.cleanlab.ai/stable/tutorials/indepth_overview.html)、[在多标注员数据中推断共识标签与标注质量](https://docs.cleanlab.ai/stable/tutorials/multiannotator.html)、[建议优先标注或重标哪些样本（主动学习）](https://github.com/cleanlab/examples/blob/master/active_learning_multiannotator/active_learning.ipynb)。

---

## 快速运行 Cleanlab 开源库

cleanlab 支持 Python 3.10+，可在 Linux、macOS 以及 Windows 环境下顺畅运行。

- [从官方文档开始](https://docs.cleanlab.ai/)！支持通过 `uv`、`pip` 或 `conda` 快速安装。
- 若从源码安装最新开发版，请参阅 [master 分支开发文档](https://docs.cleanlab.ai/master/index.html)。

**以数据为中心（Data-Centric AI）的典型迭代工作流：**
1. 在原始数据集上训练初始机器学习模型。
2. 借助该模型诊断数据问题（通过 cleanlab 算法）并改进数据集质量。
3. 在清洗优化后的数据集上重新训练相同的模型。
4. 尝试更复杂的建模技巧进一步冲刺性能上限。

大多数团队会直接从步骤 1 跳跃到步骤 4，但借助 cleanlab，你完全无需修改任何模型架构与代码就能获得显著的性能提升！
通过在步骤 2 → 4 之间持续迭代（并确保始终使用*清洗干净*的数据进行评估），持续释放模型潜能。

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/flowchart.png)

## 适用于任意模型与绝大多数机器学习任务

cleanlab 的全部功能兼容**任意数据集**与**任意模型架构**。包括但不限于：PyTorch、TensorFlow、Keras、JAX、HuggingFace、OpenAI、XGBoost、scikit-learn 等。

cleanlab 覆盖极其广泛的机器学习与深度学习任务，针对以下场景提供专用算法支持：
1. [二分类与多分类任务](https://docs.cleanlab.ai/stable/tutorials/indepth_overview.html)
2. [多标签分类](https://docs.cleanlab.ai/stable/tutorials/multilabel_classification.html)（如图像与多领域文档打标）
3. [Token 序列标注分类](https://docs.cleanlab.ai/stable/tutorials/token_classification.html)（如文本中的命名实体识别 NER）
4. [回归预测任务](https://docs.cleanlab.ai/stable/tutorials/regression.html)（预测连续数值型特征列）
5. [图像语义分割](https://docs.cleanlab.ai/stable/tutorials/segmentation.html)（包含逐像素标注的图像）
6. [目标检测](https://docs.cleanlab.ai/stable/tutorials/object_detection.html)（包含边界框 Bounding Box 标注的图像）
7. [多标注员众包数据分类](https://docs.cleanlab.ai/stable/tutorials/multiannotator.html)
8. [多标注员主动学习 (Active Learning)](https://github.com/cleanlab/examples/blob/master/active_learning_multiannotator/active_learning.ipynb)（智能推荐优先标注或重新标注哪些样本能最大化提分）
9. [离群值与异常检测 (Outlier Detection)](https://docs.cleanlab.ai/stable/tutorials/outliers.html)（精准识别分布外 OOD 异常样本）

对于其他机器学习任务，只要合理结合特征表示，cleanlab 依然能够有效提升数据集质量。
详见我们的[示例 Notebooks](https://github.com/cleanlab/examples) 与[技术博客](https://cleanlab.ai/blog/learn/)。

## 如此清爽，尽在 cleanlab

除了自动捕捉潜藏在数据中的[各类隐患与问题](https://docs.cleanlab.ai/stable/cleanlab/datalab/guide/issue_type_description.html)之外，cleanlab 还助你从容应对**标签噪声**，训练出鲁棒性极强的高质量机器学习模型。
使用示例如下：

```python
# cleanlab 兼容任意分类器 —— 无论是 PyTorch、TensorFlow、OpenAI 还是 XGBoost 等
cl = cleanlab.classification.CleanLearning(sklearn.YourFavoriteClassifier())

# 一行代码即可在任意数据集中找出数据与标签问题！
label_issues = cl.find_label_issues(data, labels)

# 自动训练抗噪版本模型，在包含脏标签的数据上依然可靠工作
cl.fit(data, labels)

# 估算若使用无标签噪声的完美数据训练本该达到的预测效果
cl.predict(test_data)

# 通用数据中心 AI 工具：量化类别层级问题与整体数据健康度
cleanlab.dataset.health_summary(labels, confident_joint=cl.confident_joint)
```

cleanlab 基于业界先进的**置信学习（Confident Learning）**理论与算法进行标签纠错，相关成果发表于权威期刊 [JAIR 论文](https://jair.org/index.php/jair/article/view/12125) 与[技术博客](https://l7.curtisnorthcutt.com/confident-learning)。部分由 cleanlab 清洗修复的公共数据集可访问 [labelerrors.com](https://labelerrors.com) 查看。

cleanlab 的核心优势：

1. **坚实理论基础** —— 具备[可证明的严格数学保证](https://arxiv.org/abs/1911.00068)，即使模型预测不完美也能精确估计标签噪声分布。
2. **高速性能** —— 原生多进程并行化设计，具备处理海量数据的卓越扩展性。
3. **极简易用** —— 一行代码即可定位错误标签、不良标注员、离群点，或训练抗噪鲁棒模型。
4. **广泛普适** —— 通用支持**[任意数据模态](https://labelerrors.com/)**（文本、图像、表格、音频等）+ **任意模型框架**（PyTorch、OpenAI、XGBoost 等）。

<br/>

![](https://raw.githubusercontent.com/cleanlab/assets/master/cleanlab/label-errors-examples.png)
<p align="center">
使用 cleanlab 在各类主流图像数据集中<a href="https://l7.curtisnorthcutt.com/label-errors">发现并纠正</a>的错误原始标签示例。
除图像数据集外，该方法同样适用于文本、音频及结构化表格数据。
</p>

## 论文引用与相关科研成果

cleanlab 建立在严谨的同行评审学术研究之上。如在学术研究中使用了本库，欢迎引用相关学术论文：

<details><summary><a href="https://arxiv.org/abs/1911.00068">Confident Learning 置信学习 (JAIR '21)</a> (<b>展开 BibTeX</b>) </summary>

    @article{northcutt2021confidentlearning,
        title={Confident Learning: Estimating Uncertainty in Dataset Labels},
        author={Curtis G. Northcutt and Lu Jiang and Isaac L. Chuang},
        journal={Journal of Artificial Intelligence Research (JAIR)},
        volume={70},
        pages={1373--1411},
        year={2021}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/1705.01936">Rank Pruning 排序剪枝 (UAI '17)</a> (<b>展开 BibTeX</b>) </summary>

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

<details><summary><a href="https://jonasmueller.org/info/LabelQuality_icml.pdf"> Label Quality Scoring 标签质量评分 (ICML '22)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{kuan2022labelquality,
        title={Model-agnostic label quality scoring to detect real-world label errors},
        author={Kuan, Johnson and Mueller, Jonas},
        booktitle={ICML DataPerf Workshop},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2210.03920"> Token 分类与实体识别中的标签错误 (NeurIPS '22)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{wang2022tokenerrors,
        title={Detecting label errors in token classification data},
        author={Wang, Wei-Chen and Mueller, Jonas},
        booktitle={NeurIPS Workshop on Interactive Learning for Natural Language Processing (InterNLP)},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2211.13895"> 多标签分类中的标签错误识别 (ICLR '23)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{thyagarajan2023multilabel,
        title={Identifying Incorrect Annotations in Multi-Label Classification Data},
        author={Thyagarajan, Aditya and Snorrason, Elías and Northcutt, Curtis and Mueller, Jonas},
        booktitle={ICLR Workshop on Trustworthy ML},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2309.00832"> ObjectLab: 目标检测中的标签错误检测 (ICML '23)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{tkachenko2023objectlab,
        title={ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data},
        author={Tkachenko, Ulyana and Thyagarajan, Aditya and Mueller, Jonas},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2307.05080"> 图像语义分割中的标签错误评估 (ICML '23)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{lad2023segmentation,
        title={Estimating label quality and errors in semantic segmentation data via any model},
        author={Lad, Vedang and Mueller, Jonas},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2305.16583"> 数值型连续数据的错误检测 (DMLR '24)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{zhou2023errors,
        title={Detecting Errors in a Numerical Response via any Regression Model},
        author={Zhou, Hang and Mueller, Jonas and Kumar, Mayank and Wang, Jane-Ling and Lei, Jing},
        booktitle={Journal of Data-centric Machine Learning Research},
        year={2024}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2207.03061"> 分布外 (OOD) 异常检测 (ICML '22)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{kuan2022ood,
        title={Back to the Basics: Revisiting Out-of-Distribution Detection Baselines},
        author={Kuan, Johnson and Mueller, Jonas},
        booktitle={ICML Workshop on Principles of Distribution Shift},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2210.06812"> CROWDLAB: 多标注员众包数据共识与质量评估 (NeurIPS '22)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{goh2022crowdlab,
        title={CROWDLAB: Supervised learning to infer consensus labels and quality scores for data with multiple annotators},
        author={Goh, Hui Wen and Tkachenko, Ulyana and Mueller, Jonas},
        booktitle={NeurIPS Human in the Loop Learning Workshop},
        year={2022}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2301.11856"> ActiveLab: 结合多标注员重标注的主动学习 (ICLR '23)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{goh2023activelab,
        title={ActiveLab: Active Learning with Re-Labeling by Multiple Annotators},
        author={Goh, Hui Wen and Mueller, Jonas},
        booktitle={ICLR Workshop on Trustworthy ML},
        year={2023}
    }

</details>

<details><summary><a href="https://arxiv.org/abs/2305.15696"> 基于 k-NN 检测数据集漂移与非独立同分布采样 (ICML '23)</a> (<b>展开 BibTeX</b>) </summary>

    @inproceedings{cummings2023drift,
        title={Detecting Dataset Drift and Non-IID Sampling via k-Nearest Neighbors},
        author={Cummings, Jesse and Snorrason, Elías and Mueller, Jonas},
        booktitle={ICML Workshop on Data-centric Machine Learning Research},
        year={2023}
    }

</details>

更多功能细节与论文引用，欢迎查阅我们的[官方博客](https://cleanlab.ai/blog/learn)。

## 更多生态资源

- [展示本库实际应用的实用示例 Notebooks 合集](https://github.com/cleanlab/examples)
- [Cleanlab 技术博客](https://cleanlab.ai/blog/learn)
- [博客文章：置信学习导论 (Introduction to Confident Learning)](https://l7.curtisnorthcutt.com/confident-learning)
- [NeurIPS 2021 论文：测试集中普遍存在的标签错误如何动摇机器学习基准](https://arxiv.org/abs/2103.14749)
- [麻省理工学院 (MIT) 以数据为中心的 AI 公开课 (MIT IAP Course)](https://dcai.csail.mit.edu/)
- [历史版本 Release Notes 更新日志](https://github.com/cleanlab/cleanlab/releases)

**想要参与贡献？** 请参阅[贡献指南 (CONTRIBUTING.md)](CONTRIBUTING.md)、[开发指南 (DEVELOPMENT.md)](DEVELOPMENT.md) 以及 [社区贡献建议与思路清单](https://github.com/cleanlab/cleanlab/wiki#ideas-for-contributing-to-cleanlab)。

**有任何疑问？** 欢迎查阅 [常见问题解答 (FAQ)](https://docs.cleanlab.ai/stable/tutorials/faq.html) 或在 [GitHub Issues](https://github.com/cleanlab/cleanlab/issues?q=is%3Aissue) 中与我们交流。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）翻译维护，最后同步更新于 2026年9月6日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
