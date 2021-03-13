# CSLU kids reading verification

This system implements a recognizer-based verification system.

To train a model using this, first make sure SpeechBrain is installed via pip.
This will change when the toolkit is released soon, but for now it can be installed
from test PyPI using the following command:

```
pip install speechbrain
```

Next, ensure you have access to the OGI kids' speech repository, say in `/path/to/cslu_kids`

You can run an experiment with:

```
cd verification/train
python train.py hparams/train.yaml --data_folder /path/to/cslu_kids
```

The different loss terms listed in `hparams/train.yaml` can be used to train a model for different tasks.

```yaml
...
ctc_weight: 0.0
align_weight: 0.0
verify_weight: 1.0
...
```

The ctc and align weight (together) can be used to train a system with reasonable alignments
between evidence and posterior, as outlined in our paper (see citation below).

The verify weight can be used to train a model to make predictions about whether each
utterance has an anomaly, based on the recording quality labels provided in the CSLU data.

## Citation

```
@inproceedings{plantinga2019towards,
  title={Towards Real-Time Mispronunciation Detection in Kids' Speech},
  author={Plantinga, Peter and Fosler-Lussier, Eric},
  booktitle={2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages={690--696},
  year={2019},
  organization={IEEE}
}
```
