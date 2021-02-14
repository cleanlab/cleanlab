import numpy as np
import os
import cleanlab
from cleanlab import baseline_methods
from cleanlab.latent_estimation import compute_confident_joint

# True, uncorrupted CIFAR-10 labels
y = np.load('cifar10_true_uncorrupted_labels.npy')

def main():


    folders = [c for c in os.listdir(base_dir) if 'noise_amount' in c]
    results = []
    for folder in sorted(folders):
        print(folder)
        psx_file = [z for z in os.listdir(base_dir + folder) if 'pyx' in z][0]
        psx = np.load(base_dir + folder + "/" + psx_file)

        # Make sure psx is the right shape
        psx = psx[:, :10]

        # Load noisy labels
        frac_zero_noise_rates = folder.split('_')[-7]
        noise_amount = folder.split('_')[-1]
        base_rfn = 'cifar10_noisy_labels__frac_zero_noise_rates__0'
        rfn = base_rfn + '.{}__noise_amount__0.{}.json'.format(
            frac_zero_noise_rates, noise_amount)
        with open(noisy_base_dir + "cifar10_noisy_labels/" + rfn, 'r') as rf:
            d = json.load(rf)
        s = np.asarray([v for k, v in d.items()])

        true_label_errors = s != y
        acc = np.sum(s == y) / len(y)
        print('accuracy of labels:', acc)

        # Benchmark methods to find label errors using using confident learning.
        # psx is the n x m matrix of cross-validated predicted probabilities
        # s is the array of given noisy labels

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

        # Create folders and store clean label np.array bool masks for training.
        clean_labels = {
            'conf_joint_only': ~baseline_conf_joint_only,
            'pruned_argmax': ~baseline_argmax,
            'cl_pbc': ~baseline_cl_pbc,
            'cl_pbnr': ~baseline_cl_pbnr,
            'cl_both': ~baseline_cl_both,
        }
        for name, labels in clean_labels.items():
            new_folder = base_dir + folder + "/train_pruned_" + name + "/"
            try:
                os.mkdir(new_folder)
            except FileExistsError:
                pass
            np.save(new_folder + "train_mask.npy", labels)
        print()


if __name__ == '__main__':
    main()