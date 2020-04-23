# State-of-the-art learning with noisy labels on CIFAR-10.

All data needed to reproduce state-of-the-art results is available in this repo: [cgnorthcutt/confidentlearning-reproduce](https://github.com/cgnorthcutt/confidentlearning-reproduce/tree/master/cifar10).
All the code needed is available here, in the `cleanlab` package.
This code can be used to achieve state-of-the-art (as of Feb. 2020) for learning with noisy labels on CIFAR-10.

The main procedure is simple:
1. Compute cross-validated predicted probabilities.
2. Use `cleanlab` to find the label errors in CIFAR-10.
3. Remove errors and train on cleaned data via [Co-Teaching](https://github.com/cgnorthcutt/cleanlab/blob/4245cde2f177cdb999b7216fef67d3fe6425982d/examples/cifar10/cifar10_train_crossval.py#L139).

## Step-by-step: finding label errors and state-of-the-art test accuracy.


### 1. A PyTorch-ready version of CIFAR-10 dataset

To facilitate these computations, a PyTorch-prepared version of the CIFAR-10 dataset is available [here for download](https://github.com/cgnorthcutt/confidentlearning-reproduce/tree/master/cifar10/dataset): [`cgnorthcutt/confidentlearning-reproduce/cifar10/dataset`](https://github.com/cgnorthcutt/confidentlearning-reproduce/tree/master/cifar10/dataset). The dataset was prepared by creating `train/` and `test/` directories and organizing their images into folders by class.

### 2. Compute the cross-validated predicted probabilities

The code below shows how to compute cross-validated predicted probabilities for a given labels file `LABELS_PATH.json`.

```bash
$ python3 cifar10_train_crossval.py \
    -a resnet50 --gpu 0 --cvn 4 --cv 0  \
    --train-labels LABELS_PATH.json CIFAR10_PATH
$ python3 cifar10_train_crossval.py \
    -a resnet50 --gpu 1 --cvn 4 --cv 1  \
  --train-labels LABELS_PATH.json CIFAR10_PATH
$ python3 cifar10_train_crossval.py \
    -a resnet50 --gpu 2 --cvn 4 --cv 2  \
  --train-labels LABELS_PATH.json CIFAR10_PATH
$ python3 cifar10_train_crossval.py \
    -a resnet50 --gpu 3 --cvn 4 --cv 3  \
  --train-labels LABELS_PATH.json CIFAR10_PATH
```

where an example of `LABELS_PATH.json` might be `/home/cgn/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.4.json` and
`CIFAR10_PATH` is the absolute path to the CIFAR dataset (it should be a directory containing a `train/` and `test/` folder).

Each of the above commands will output a `.npy` file with 1/4 of the predicted probabilities on the dataset.


### 3. Combining the cv fold partial `.npy` outputs to get `psx`.

Each cross-validation fold outputs only 1/4 of the predicted probabilities. We need to combine them. We can do this easily:

```bash
$ python3 imagenet_train_crossval.py --cvn 4 --combine-folds CIFAR10_PATH
```

Make sure you run this in the same path as all the .npy files containing the predicted probabilities for each fold.

`psx` stands for prob(s|x), the predicted probability of the noisy label `s` for every example `x`. This should be a `n` (number of examples) x `m` (number of classes) matrix.

#### Pre-computed `psx` for every noise / sparsity condition

If you want to save time, I've already done the above step for you. You can download the `psx` predicted probabilities for all CIFAR-10 training examples computed using four-fold cross-validation with a ResNet50 architecture for every noise / sparsity condition. The code above was used exactly. 

 * Noise: 0% | Sparsity: 0% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_0__noise_amount__0_0/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 20% | Sparsity: 0% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_0__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 40% | Sparsity: 0% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_0__noise_amount__0_4/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 70% | Sparsity: 0% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_0__noise_amount__0_6/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 20% | Sparsity: 20% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_2__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 40% | Sparsity: 20% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_2__noise_amount__0_4/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 70% | Sparsity: 20% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_2__noise_amount__0_6/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 20% | Sparsity: 40% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_4__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 40% | Sparsity: 40% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_4__noise_amount__0_4/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 70% | Sparsity: 40% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_4__noise_amount__0_6/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 20% | Sparsity: 60% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_6__noise_amount__0_2/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 40% | Sparsity: 60% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_6__noise_amount__0_4/cifar10__train__model_resnet50__pyx.npy)]
 * Noise: 70% | Sparsity: 60% | [[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_noisy_labels__frac_zero_noise_rates__0_6__noise_amount__0_6/cifar10__train__model_resnet50__pyx.npy)]


### 4. Use confident learning to find the label errors.

Now that we have the predicted probabilities, and of course, we have the noisy labels. We can use confident learning via the `cleanlab` package to find the label errors.

```python3
# cleanlab code for computing the 5 confident learning methods.
# psx is the n x m matrix of cross-validated predicted probabilities
# s is the array of noisy labels

# Method: C_{\tilde{y}, y^*}
label_error_mask = np.zeros(len(s), dtype=bool)
label_error_indices = compute_confident_joint(
    s, psx, return_indices_of_off_diagonals=True
)[1]
for idx in label_error_indices:
    label_error_mask[idx] = True
baseline_conf_joint_only = label_error_mask

# Method: C_confusion
baseline_argmax = baseline_methods.baseline_argmax(psx, s)

# Method: CL: PBC
baseline_cl_pbc = cleanlab.pruning.get_noise_indices(
            s, psx, prune_method='prune_by_class')

# Method: CL: PBNR
baseline_cl_pbnr = cleanlab.pruning.get_noise_indices(
            s, psx, prune_method='prune_by_noise_rate')

# Method: CL: C+NR
baseline_cl_both = cleanlab.pruning.get_noise_indices(
            s, psx, prune_method='both')
```

We compute all five of the above methods for finding label errors, for every set of noisy labels across all conditions. The complete code is available [here](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/cifar10_benchmarking.ipynb) (see the section entitled "Setting up training experiments.").


### Precomputed label errors for CIFAR-10 train set

Using the `psx` predicted probabilities above as input, I used [**`cleanlab`**](https://pypi.org/project/cleanlab/) to pre-compute the label errors for every confident learning method in [the CL paper](https://arxiv.org/abs/1911.00068), for every noise and sparsity setting. The outputs are boolean numpy arrays. They are ordered in the same order as the examples when loaded using [`torch.utils.data.dataloader`](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html). The PyTorch-prepared CIFAR dataset is available here for download: [`cifar10/dataset`](https://github.com/cgnorthcutt/confidentlearning-reproduce/tree/master/cifar10/dataset). If you load this dataset in PyTorch, indices will match exactly with the label error masks below.

Column headers are formatted as: *<sparsity * 10>\_<noise * 10>*.


| METHOD | 0_2 | 2_2 | 4_2 | 6_2 | 0_4 | 2_4 | 4_4 | 6_4 | 0_7 | 2_7 | 4_7 | 6_7 |
|--------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| **C_confusion** | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_2/train_pruned_argmax/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_2/train_pruned_argmax/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_2/train_pruned_argmax/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_2/train_pruned_argmax/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_4/train_pruned_argmax/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_4/train_pruned_argmax/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_4/train_pruned_argmax/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_4/train_pruned_argmax/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_7/train_pruned_argmax/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_7/train_pruned_argmax/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_7/train_pruned_argmax/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_7/train_pruned_argmax/train_mask.npy) |
| **![formula](https://render.githubusercontent.com/render/math?math=C_{\tilde{y},y^*})** | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_2/train_pruned_conf_joint_only/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_2/train_pruned_conf_joint_only/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_2/train_pruned_conf_joint_only/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_2/train_pruned_conf_joint_only/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_4/train_pruned_conf_joint_only/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_4/train_pruned_conf_joint_only/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_4/train_pruned_conf_joint_only/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_4/train_pruned_conf_joint_only/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_7/train_pruned_conf_joint_only/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_7/train_pruned_conf_joint_only/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_7/train_pruned_conf_joint_only/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_7/train_pruned_conf_joint_only/train_mask.npy) |
| **CL: PBC** |  [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_2/train_pruned_cl_pbc/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_2/train_pruned_cl_pbc/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_2/train_pruned_cl_pbc/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_2/train_pruned_cl_pbc/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_4/train_pruned_cl_pbc/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_4/train_pruned_cl_pbc/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_4/train_pruned_cl_pbc/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_4/train_pruned_cl_pbc/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_7/train_pruned_cl_pbc/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_7/train_pruned_cl_pbc/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_7/train_pruned_cl_pbc/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_7/train_pruned_cl_pbc/train_mask.npy) |
| **CL: PBNR** |  [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_2/train_pruned_cl_pbnr/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_2/train_pruned_cl_pbnr/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_2/train_pruned_cl_pbnr/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_2/train_pruned_cl_pbnr/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_4/train_pruned_cl_pbnr/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_4/train_pruned_cl_pbnr/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_4/train_pruned_cl_pbnr/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_4/train_pruned_cl_pbnr/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_7/train_pruned_cl_pbnr/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_7/train_pruned_cl_pbnr/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_7/train_pruned_cl_pbnr/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_7/train_pruned_cl_pbnr/train_mask.npy) |
| **CL: C+NR** |  [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_2/train_pruned_cl_both/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_2/train_pruned_cl_both/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_2/train_pruned_cl_both/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_2/train_pruned_cl_both/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_4/train_pruned_cl_both/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_4/train_pruned_cl_both/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_4/train_pruned_cl_both/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_4/train_pruned_cl_both/train_mask.npy) |[LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/0_7/train_pruned_cl_both/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/2_7/train_pruned_cl_both/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/4_7/train_pruned_cl_both/train_mask.npy) | [LINK](https://github.com/cgnorthcutt/confidentlearning-reproduce/blob/master/cifar10/confidentlearning_and_coteaching/results/6_7/train_pruned_cl_both/train_mask.npy) |


### 5. Final training

The noise masks have already been precomputed. Here is an example of how to run Confident Learning training with Co-Teaching on labels with 40% label noise (noise is asymmetric and in this example we'll look at label noise with 40% sparsity);

```bash
{ time python3 ~/cgn/cleanlab/examples/cifar10/cifar10_train_crossval.py \
	--coteaching \
    	--seed 1 \
	--batch-size 128 \
	--lr 0.001 \
	--epochs 250 \
	--turn-off-save-checkpoint \
	--train-labels /home/cgn/cifar10/cifar10_noisy_labels/cifar10_noisy_labels__frac_zero_noise_rates__0.4__noise_amount__0.4.json \
	--gpu 0 \
	--dir-train-mask /home/cgn/cifar10/4_4/train_pruned_conf_joint_only/train_mask.npy \
	/PATH/TO/CIFAR10/DATASET/ ; \
} &> out_4_4.log &
tail -f out_4_4.log;
```

This bash command does a few things:
1. it wraps inside of the bash `time` function so we can get total training time.
2. It stores the output in a log file so we can see the resulting test accuracy for each epoch later.
3. It uses `tail -f` to output while running the process in the background and storing the file.

Additional information about the parameters used in the Python command:
* --coteaching uses the CoTeaching algorithm for training.
* --seed 1 makes results reproducible, although similar results are obtained without it.
* --turn-off-save-checkpoint is not necessary, it just prevents the code from saving the large 50MB model file every epoch.
* --gpu 0 chooses GPU 0. If you have multiple gpus, select whichever GPU you like.
* --train-labels is the path to a json file that maps image ids to noisy labels.
* --dir-train-mask is a npy file storing a boolean mask for the CLEANED dataset. We computed this earlier.

For the directories used above, you'll need to first get the data from [cgnorthcutt/confidentlearning-reproduce](https://github.com/cgnorthcutt/confidentlearning-reproduce/tree/master/cifar10) then update the paths.




## License

Copyright (c) 2017-2020 Curtis Northcutt. Released under the MIT License. See [LICENSE](https://github.com/cgnorthcutt/cleanlab/blob/master/LICENSE) for details.
