from __future__ import print_function

import os
import sys

import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
import random
from sklearn.mixture import GaussianMixture as GMM


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


class Color:
    GRAY = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    CRIMSON = 38


def colorize(num, string, bold=False, highlight=False):
    assert isinstance(num, int)
    attr = []
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def colorprint(colorcode, text, o=sys.stdout, bold=False):
    o.write(colorize(colorcode, text, bold=bold))


def generate_image_patches_db(in_directory, out_directory, patch_size=64):
    out_directory = out_directory + '_' + str(patch_size)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    total = 2688
    count = 0
    splits = ["train", "test"]
    for split_dir in splits:
        if not os.path.exists(os.path.join(out_directory, split_dir)):
            os.makedirs(os.path.join(out_directory, split_dir))
        for class_dir in os.listdir(os.path.join(in_directory, split_dir)):

            if not os.path.exists(os.path.join(out_directory, split_dir, class_dir)):
                os.makedirs(os.path.join(out_directory, split_dir, class_dir))

            for imname in os.listdir(os.path.join(in_directory, split_dir, class_dir)):
                count += 1
                print('Processed images: ' + str(count) + ' / ' + str(total), end='\r')
                im = Image.open(os.path.join(in_directory, split_dir, class_dir, imname))

                patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size),
                                                   max_patches=int(np.asarray(im).shape[0] / patch_size) ** 2)
                for i, patch in enumerate(patches):
                    patch = Image.fromarray(patch)
                    patch.save(
                        os.path.join(out_directory, split_dir, class_dir, imname.split(',')[0] + '_' + str(i) + '.jpg'))
    print('\n')


def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
            - Q_xx_2
            - Q_sum * gmm.means_ ** 2
            + Q_sum * gmm.covariances_
            + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def train_fv(features, descriptors):
    num_gmm = 16

    # Fit the GMM
    indices = random.sample(range(0, features.shape[0]), num_gmm * 1500)
    sample = features[indices, :]

    print('Fitting the GMM')
    gmm = GMM(n_components=num_gmm, covariance_type='diag')
    gmm.fit(sample)
    print('GMM fitted')

    # Obtain the Fisher Vectors for the training dataset
    train_descriptors_FV = []
    for train_descriptor in descriptors:
        start = 0
        length = train_descriptor.shape[0]
        stop = start + length
        train_descriptor = train_descriptor[:, :descriptors[0].shape[1]]
        start = stop
        train_descriptor_fisher = fisher_vector(train_descriptor, gmm)
        train_descriptors_FV.append(train_descriptor_fisher)
        image_fv = np.vstack(train_descriptors_FV)
        image_fv = np.sign(image_fv) * np.abs(image_fv) ** 0.5
        norms = np.sqrt(np.sum(image_fv ** 2, 1))
        train_FV = image_fv / norms.reshape(-1, 1)

    return train_FV


def intersection_kernel(A,B):
    """
    Histogram intersection kernel function defined as:
    K_int(A, B) = SUM_i(min(a_i, b_i))

    Parameters
    ---------
    A: histogram
    B: histogram

    Return
    ---------
    k_int: kernel function
    """

    # Create the kernel parameter
    k_int = np.zeros((A.shape[0], B.shape[0]))

    # Shape of the histograms
    size = A.shape[1]

    # Iterate through bins
    for idx in range(size):
        bin_A = A[:, idx].reshape(-1,1)
        bin_B = B[:, idx].reshape(-1,1)
        k_int += np.minimum(bin_A, bin_B.T)

    return k_int