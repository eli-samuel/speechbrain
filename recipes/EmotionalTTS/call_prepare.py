#!/usr/bin/env python3
"""Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality)
with an ECAPA-TDNN model.

To run this recipe, do the following:

> python call_prepare.py hparams/train.yaml --data_folder /path/to/IEMOCAP

Authors
 * Eli Samuel 2023
"""

import os
import sys
import csv
import speechbrain as sb
import torch
from torch.utils.data import DataLoader
from enum import Enum, auto
from tqdm.contrib import tqdm
from hyperpyyaml import load_hyperpyyaml


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.encode_label_torch(emo)
        yield emo_encoded
    print("a")
    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    print(data_info)
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    print(len(datasets["train"]))
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="emo",
    )

    print("b")
    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from iemocap_prepare import prepare_iemocap  # noqa E402

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_iemocap,
            kwargs={
                "data_original": hparams["data_folder"],
                "save_json_train": hparams["train_annotation"],
                "save_json_valid": hparams["valid_annotation"],
                "save_json_test": hparams["test_annotation"],
                "split_ratio": hparams["split_ratio"],
                "different_speakers": hparams["different_speakers"],
                "test_spk_id": hparams["test_spk_id"],
                "seed": hparams["seed"],
            },
        )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)
    
    from compute_speaker_embeddings import compute_speaker_embeddings

    sb.utils.distributed.run_on_main(
        compute_speaker_embeddings,
        kwargs={
            "input_filepaths": [
                hparams["train_json"],
                hparams["valid_json"],
                hparams["test_json"],
            ],
            "output_file_paths": [
                hparams["train_speaker_embeddings_pickle"],
                hparams["valid_speaker_embeddings_pickle"],
                hparams["test_speaker_embeddings_pickle"],
            ],
            "data_folder": hparams["data_folder"],
            "spk_emb_encoder_path": hparams["spk_emb_encoder"],
            "spk_emb_sr": hparams["spk_emb_sample_rate"],
            "mel_spec_params": {
                "custom_mel_spec_encoder": hparams["custom_mel_spec_encoder"],
                "sample_rate": hparams["spk_emb_sample_rate"],
                "hop_length": hparams["hop_length"],
                "win_length": hparams["win_length"],
                "n_mel_channels": hparams["n_mel_channels"],
                "n_fft": hparams["n_fft"],
                "mel_fmin": hparams["mel_fmin"],
                "mel_fmax": hparams["mel_fmax"],
                "mel_normalized": hparams["mel_normalized"],
                "power": hparams["power"],
                "norm": hparams["norm"],
                "mel_scale": hparams["mel_scale"],
                "dynamic_range_compression": hparams[
                    "dynamic_range_compression"
                ],
            },
            "device": run_opts["device"],
        },
    )