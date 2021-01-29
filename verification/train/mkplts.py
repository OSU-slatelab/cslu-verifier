#!/usr/bin/env python

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cslu_prepare import prepare_cslu
from train import ASR, dataio_prep

mpl.use("Agg")
mpl.rcParams.update({"font.size": 14})

if __name__ == "__main__":

    hparams_file = sys.argv[1]
    utterance_id = sys.argv[2]
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    prepare_voicebank(
        data_folder=hparams["data_folder"], save_folder=hparams["save_folder"]
    )

    datasets, tokenizer = dataio_prep(hparams)

    # Load Brain class
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
    )

    # Print single utterance's posteriorgram
    # utterance = datasets["valid"].get_utterance()
    # output = asr_brain.compute_forward(utterance)
    # posterior, pos_energy = prepare_posterior(output)
    # print_posteriorgram(
    #     filename=utterance_id + ".pdf",
    #     ind2lab=tokenizer.ind2lab,
    #     posterior=posterior,
    #     energy=energy,
    # )


def prepare_posterior(outputs):
    """Prepare for output, including converting to numpy"""
    posterior = torch.nn.functional.softmax(outputs["pout"], dim=-1)
    posterior = posterior.detach().cpu().numpy()
    pos_energy = outputs["energy"].detach().cpu().numpy()
    return posterior, pos_energy


def printSpec(array, filename, xaxisRange=None, axes="on"):
    """ Print a spectrogram to a file """

    if xaxisRange:
        array = np.flipud(array[xaxisRange[0] : xaxisRange[1]].T)
        extent = [xaxisRange[0] / 33.0, xaxisRange[1] / 33.0, 0, 8]
    else:
        array = np.flipud(array.T)
        extent = [0, array.shape[1] / 33.0, 0, 8]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(
        array, cmap=plt.cm.jet, interpolation="none", extent=extent, aspect=1.0 / 7,
    )

    if axes == "on":
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        fig.savefig(filename, format="pdf", bbox_inches="tight")
    else:
        ax.axis("off")
        fig.savefig(filename, format="pdf", bbox_inches=0)

    plt.close(fig)


def print_posteriorgram(
    filename, index2label, posterior, energy=None, cutoff=0.1, xAxisRange=None
):
    """
    Generate a posterior output.
    """
    print("Outputting posterior to file %s." % filename)

    # Exclude blank index
    posterior = posterior[:, :-1]

    # Indexes where the max posterior is greater than some cutoff
    greater_indexes = np.where(np.amax(posterior, axis=1) > cutoff)[0]

    # Index of the label at the positions greater than 0.1
    label_indexes = np.argmax(posterior[greater_indexes], axis=1)

    # Remove duplicates
    _, idx = np.unique(label_indexes, return_index=True)
    label_indexes = label_indexes[np.sort(idx)]

    # Only plot those lines that have some activation
    posterior = posterior[:, label_indexes]

    if xAxisRange is not None:
        posterior = posterior[xAxisRange[0] : xAxisRange[1]]
        if energy is not None:
            energy = energy[xAxisRange[0] : xAxisRange[1]]
    else:
        xAxisRange = (0, len(posterior))

    # Plot our figure
    fig = plt.figure()
    xRange = np.arange(xAxisRange[0], xAxisRange[1]) / 33.0
    ax = fig.add_subplot(111, aspect=1.5)
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=2)
    ax.plot(xRange, posterior)
    ax.legend(labels=[index2label[i] for i in label_indexes])

    # Print energy if added
    if energy is not None:
        energy -= np.mean(energy)
        energy /= np.max(np.abs(energy)) * 3
        energy += 0.5
        ax.plot(xRange, energy, linewidth=1, color="k", linestyle="--")
        # ax.plot(xRange, np.mean(energy), linewidth=1, color='k', linestyle=':')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Posterior")
    fig.savefig(filename, format="pdf", bbox_inches="tight")

    plt.close(fig)
