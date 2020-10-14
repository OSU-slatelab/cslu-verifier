#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 14})
import matplotlib.pyplot as plt


def printSpec(array, filename, xaxisRange=None, axes='on'):
    """ Print a spectrogram to a file """

    if xaxisRange:
        array = np.flipud(array[xaxisRange[0]:xaxisRange[1]].T)
        extent = [xaxisRange[0] / 33., xaxisRange[1] / 33., 0, 8]
    else:
        array = np.flipud(array.T)
        extent = [0, array.shape[1] / 33., 0, 8]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(array,cmap=plt.cm.jet, interpolation='none', extent=extent, aspect=1./7)

    if axes == 'on':
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        fig.savefig(filename, format='pdf', bbox_inches='tight')
    else:
        ax.axis('off')
        fig.savefig(filename, format='pdf', bbox_inches=0)

    plt.close(fig)


def print_posteriorgram(
    filename, index2label, posterior, energy=None, cutoff=0.1, xAxisRange=None
):
    """
    Generate a posterior output.
    """
    print("Outputting posterior to file %s." % filename)

    # Exclude blank index
    posterior = posterior[:,:-1]

    # Indexes where the max posterior is greater than some cutoff
    greater_indexes = np.where(np.amax(posterior, axis=1) > cutoff)[0]

    # Index of the label at the positions greater than 0.1
    label_indexes = np.argmax(posterior[greater_indexes], axis=1)

    # Remove duplicates
    _, idx = np.unique(label_indexes, return_index=True)
    label_indexes = label_indexes[np.sort(idx)]

    # Only plot those lines that have some activation
    posterior = posterior[:,label_indexes]

    if xAxisRange is not None:
        posterior = posterior[xAxisRange[0]:xAxisRange[1]]
        if energy is not None:
            energy = energy[xAxisRange[0]:xAxisRange[1]]
    else:
        xAxisRange = (0, len(posterior))


    # Plot our figure
    fig = plt.figure()
    xRange = np.arange(xAxisRange[0], xAxisRange[1]) / 33.
    ax = fig.add_subplot(111, aspect=1.5)
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=2)
    ax.plot(xRange, posterior)
    ax.legend(labels=[index2label[i] for i in label_indexes])

    # Print energy if added
    if energy is not None:
        energy -= np.mean(energy)
        energy /= np.max(np.abs(energy)) * 3
        energy += 0.5
        ax.plot(xRange, energy, linewidth=1, color='k', linestyle='--')
        #ax.plot(xRange, np.mean(energy), linewidth=1, color='k', linestyle=':')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Posterior")
    fig.savefig(filename, format='pdf', bbox_inches='tight')

    plt.close(fig)
