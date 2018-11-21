# Finding Label Errors in the 2012 ImageNet validation dataset.

## To find the label errors yourself, you'll need to:
#### Download the 2012 ImageNet validation set here:
http://places.csail.mit.edu/imagenet/ilsvrc2012_val.tar

#### Move ImageNet validation images to labeled subfolders using Soumith's script here:
https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## If you just want a list of all the label errors in 2012 ImageNet validation set:

#### The top 5000 predicted label errors are available as a text file here:
https://github.com/cgnorthcutt/cleanlab/blob/master/examples/finding_ImageNet_label_errors/imagenet_val_label_errors.txt

This text file contains a list of 5000 Python dicts, each representing a predicted label error.  
The first two dicts in the list, look like:

```python
# Ordered by prob_max_label - prob_given_label
[{u'given_label': u'bucket',
  u'id_str': u'n02909870/ILSVRC2012_val_00006594.JPEG',
  u'predicted_label': u'baseball',
  u'prob_given_label': 1.1067604e-10,
  u'prob_max_label': 1.0},
 {u'given_label': u'assault rifle',
  u'id_str': u'n02749479/ILSVRC2012_val_00040844.JPEG',
  u'predicted_label': u'bearskin',
  u'prob_given_label': 4.8315085e-12,
  u'prob_max_label': 1.0},
  ...
 ]
```

The dicts are ordered from top to bottom by confidence of being a label error. Computationally, this means they are sorted by prob_max_label - prob_given_label.

#### You can download a smaller file containing only the unique id's of the label errors here:
https://github.com/cgnorthcutt/cleanlab/blob/master/examples/finding_ImageNet_label_errors/imagenet_val_label_errors_unique_id_only.txt
