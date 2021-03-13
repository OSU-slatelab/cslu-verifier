#!/usr/bin/env python3
"""Recipe for finding "invalid" or abnormal utterances in kids' speech.

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/CSLU

Authors
 * Peter Plantinga 2021
"""
import sys
import torch
import speechbrain as sb
from cslu_prepare import prepare_cslu
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        chars, char_lens = batch.tokens

        # Adding environmental corruption if specified (i.e., noise+rev)
        if hasattr(self.hparams, "env_corrupt") and stage == sb.Stage.TRAIN:
            wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        # Adding time-domain SpecAugment if specified
        if hasattr(self.hparams, "augmentation") and stage == sb.Stage.TRAIN:
            wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats, wav_lens)
        out = self.modules.recognizer(feats)

        # Verifier prediction
        predictions = {}
        if self.hparams.verify_weight > 0:
            predictions["verify"] = self.modules.verifier(out, chars)

        # CTC prediction
        predictions["out"] = self.modules.recognizer_output(out)
        predictions["pout"] = self.hparams.log_softmax(predictions["out"])

        # reshape feats (energies) to be same shape as pout
        batch, length, _ = predictions["pout"].shape
        predictions["energy"] = feats[:, : length * 4].reshape(batch, length, -1)
        predictions["energy"] = predictions["energy"].sum(dim=-1)

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        wavs, wav_lens = batch.sig
        chars, char_lens = batch.tokens
        pout, out = predictions["pout"], predictions["out"]

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corrupt"):
            chars = torch.cat([chars, chars], dim=0)
            char_lens = torch.cat([char_lens, char_lens], dim=0)

        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(predictions["pout"], wav_lens)
            self.cer_metrics.append(
                ids=batch.id,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.tokenizer.decode_ndim,
            )

            # Compute CER for only clean samples
            clean_samples = batch.verify == 1
            clean_seq = [sequence[i] for i, s in enumerate(clean_samples) if s]
            clean_chars = chars[clean_samples]
            clean_lens = char_lens[clean_samples]
            clean_ids = [batch.id[i] for i, s in enumerate(clean_samples) if s]
            self.clean_cer_metrics.append(
                ids=clean_ids,
                predict=clean_seq,
                target=clean_chars,
                target_len=clean_lens,
                ind2lab=self.tokenizer.decode_ndim,
            )

        if self.hparams.verify_weight > 0:
            label = (batch.verify != 1).int().to(self.device)
            label = label.unsqueeze(1)

            if stage != sb.Stage.TRAIN:
                self.verify_metrics.append(ids, prediction, label)

        # Compute each loss if requested
        loss = 0

        if self.hparams.ctc_weight > 0:
            ctc_loss = self.hparams.ctc_loss(pout, chars, wav_lens, char_lens)
            loss += self.hparams.ctc_weight * ctc_loss

        if self.hparams.align_weight > 0:
            align_loss = self.alignment_loss(out, predictions["energy"], wav_lens)
            loss += self.hparams.align_weight * align_loss

        if self.hparams.verify_weight > 0:
            verify_loss = self.hparams.bce_loss(predictions["verify"], label)
            loss += self.hparams.verify_weight * verify_loss

        return loss

    def alignment_loss(self, prediction, energies, length, blank_idx=-1):
        abs_length = sb.dataio.dataio.relative_time_to_absolute(energies, length, 1)
        avg_energies = energies.sum(dim=1, keepdim=True)
        avg_energies /= abs_length.unsqueeze(1)

        # Factor to multiply energy score by (for all but blank id)
        factor = (energies < avg_energies).int() * 2 - 1
        output_count = prediction.size(-1)
        factor = factor.unsqueeze(-1).repeat(1, 1, output_count)

        # Flip factor for blank index
        factor[:, :, blank_idx] *= -1

        # Function to compute loss using prediction and avg energy
        def loss_fn(prediction, factor):
            return torch.nn.functional.softmax(prediction, dim=-1) * factor

        return sb.nnet.losses.compute_masked_loss(loss_fn, prediction, factor, length)

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.cer_metrics = self.hparams.cer_computer()
            self.clean_cer_metrics = self.hparams.cer_computer()
            if self.hparams.verify_weight > 0:
                self.verify_metrics = self.hparams.verify_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            cer = self.cer_metrics.summarize("error_rate")
            clean_cer = self.clean_cer_metrics.summarize("error_rate")
            stage_stats = {
                "loss": stage_loss,
                "CER": cer,
                "CleanCER": clean_cer,
            }
            max_keys = []
            if self.hparams.verify_weight > 0:
                f1 = self.verify_metrics.summarize("F-score", threshold=0.5)
                stage_stats["F1"] = f1 * 100
                max_keys = ["F1"]

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(cer)
            epoch_stats = {"epoch": epoch, "lr": old_lr}
            self.hparams.train_logger.log_stats(
                epoch_stats, {"loss": self.train_loss}, stage_stats
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["CER"], max_keys=max_keys,
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, "w") as w:
                self.cer_metrics.write_stats(w)


def dataio_prep(hparams):
    """Creates datasets and loading pipelines"""

    tokenizer = sb.dataio.encoder.CTCTextEncoder()

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    @sb.utils.data_pipeline.takes("char")
    @sb.utils.data_pipeline.provides("tokens_list", "tokens")
    def text_pipeline(char):
        tokens_list = char.strip().split()
        yield tokens_list
        tokens = tokenizer.encode_sequence(tokens_list)
        yield torch.LongTensor(tokens)

    data = {}
    for dataset in ["train", "valid", "test"]:
        data[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=["id", "sig", "tokens", "verify"],
        ).filtered_sorted(sort_key="length")

    tokenizer.update_from_didataset(data["train"], output_key="tokens_list")

    return data, tokenizer


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    run_on_main(
        prepare_cslu,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["data_folder"],
        },
    )

    datasets, tokenizer = dataio_prep(hparams)

    # Create brain object for training
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.tokenizer = tokenizer

    asr_brain.fit(
        epoch_counter=asr_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_loader_options"],
        valid_loader_kwargs=hparams["valid_loader_options"],
    )

    asr_brain.evaluate(
        test_set=datasets["test"],
        min_key="CleanCER",
        test_loader_kwargs=hparams["test_loader_options"],
    )
