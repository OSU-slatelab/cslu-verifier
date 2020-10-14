#!/usr/bin/env python3
import os
import sys
import torch
import speechbrain as sb
from mkplts import print_posteriorgram


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, x, stage):
        ids, wavs, wav_lens = x
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Adding environmental corruption if specified (i.e., noise+rev)
        if hasattr(self.hparams, "env_corrupt") and stage == "train":
            wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
            wavs = torch.cat([wavs, wavs_noise], dim=0)
            wav_lens = torch.cat([wav_lens, wav_lens])

        # Adding time-domain SpecAugment if specified
        if hasattr(self.hparams, "augmentation"):
            wavs = self.hparams.augmentation(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.hparams.normalize(feats)
        out = self.modules.model(feats)

        # Verifier prediction
        #verifier_prediction = self.modules.verifier(out)
        
        # CTC prediction
        out = self.modules.output(out)
        pout = self.hparams.log_softmax(out)

        # reshape feats (energies) to be same shape as pout
        batch, length, _ = pout.shape
        energy = feats[:, :length * 4].reshape(batch, length, -1).sum(dim=-1)

        if hasattr(self.hparams, "print_id") and self.hparams.print_id in ids:
            index = ids.index(self.hparams.print_id)
            posterior = torch.nn.functional.softmax(out[index], dim=-1)
            posterior = posterior.detach().cpu().numpy()
            pos_energy = energy[index].detach().cpu().numpy()
            filename = self.hparams.print_id + ".pdf"
            print_posteriorgram(
                filename,
                self.hparams.ind2lab,
                posterior,
                pos_energy,
                #xAxisRange=(20,120),
            )
            sys.exit()

        return pout, wav_lens, out, energy #, verifier_prediction 

    def compute_objectives(self, predictions, targets, stage, verify=None):
        #pout, pout_lens, out, energies, verifier_prediction = predictions
        pout, pout_lens, out, energies = predictions
        ids, chars, char_lens = targets
        chars, char_lens = chars.to(self.device), char_lens.to(self.device)
        if verify is not None:
            ids, verify, _ = verify
            verify = torch.tensor([int(v[0]) for v in verify])

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corrupt"):
            chars = torch.cat([chars, chars], dim=0)
            char_lens = torch.cat([char_lens, char_lens], dim=0)
        
        if stage != sb.Stage.TRAIN or self.hparams.simple_verify_alg:
            sequence = sb.decoders.ctc_greedy_decode(pout, pout_lens)
            self.cer_metrics.append(
                ids, sequence, chars, None, char_lens, self.hparams.ind2lab
            )

            # Compute CER for only clean samples
            clean_samples = verify == 1
            clean_seq = [sequence[i] for i, s in enumerate(clean_samples) if s]
            clean_chars = chars[clean_samples]
            clean_lens = char_lens[clean_samples]
            clean_ids = [ids[i] for i, s in enumerate(clean_samples) if s]
            self.clean_cer_metrics.append(
                clean_ids, clean_seq, clean_chars, None, clean_lens, self.hparams.ind2lab
            )
            
        # Generate verify predictions and labels
        if self.hparams.simple_verify_alg:
            # Generate reading verification prediction (> 1 error)
            scores = self.cer_metrics.scores[-len(ids):]
            errs = torch.tensor([s["num_edits"] for s in scores])
            prediction = (errs > 2).int()

        if stage != sb.Stage.TRAIN or self.hparams.verify_weight > 0:
            # Exclude sentences that don't have a verify label
            hasverify = verify != 0
            ids = [ids[i] for i, h in enumerate(hasverify) if h]
            prediction = prediction[hasverify]

            # Verify score of 1 are pronounced correctly, otherwise not
            label = (verify[hasverify] != 1).int()
            self.verify_metrics.append(ids, prediction, label)

        # Compute each loss if requested
        loss = 0

        if self.hparams.ctc_weight > 0:
            ctc_loss = self.hparams.ctc_loss(pout, chars, pout_lens, char_lens)
            loss += self.hparams.ctc_weight * ctc_loss

        if self.hparams.align_weight > 0:
            align_loss = self.alignment_loss(out, energies, pout_lens)
            loss += self.hparams.align_weight * align_loss

        if self.hparams.verify_weight > 0:
            verify_loss = self.hparams.bce_loss(prediction, label)
            loss += self.hparams.verify_weight * verify_loss

        return loss
    
    def alignment_loss(self, prediction, energies, length, blank_idx=-1):
        abs_length = sb.data_io.relative_time_to_absolute(energies, length, 1)
        avg_energies = energies.sum(dim=1, keepdim=True)
        avg_energies /= abs_length.unsqueeze(1)

        # Factor to multiply energy score by (for all but blank id)
        factor = (energies < avg_energies).int() * 2 - 1
        output_count = prediction.size(-1)
        factor = factor.unsqueeze(-1).repeat(1, 1, output_count)

        # Flip factor for blank index
        factor[blank_idx] *= -1

        # Function to compute loss using prediction and avg energy
        def loss_fn(prediction, factor):
            return torch.nn.functional.softmax(prediction, dim=-1) * factor

        return sb.nnet.compute_masked_loss(loss_fn, prediction, factor, length)

    def evaluate_batch(self, batch, stage):
        inputs, targets, verify = batch
        out = self.compute_forward(inputs, stage)
        loss = self.compute_objectives(out, targets, stage, verify)
        return loss.detach().cpu()

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.cer_metrics = self.hparams.cer_computer()
            self.clean_cer_metrics = self.hparams.cer_computer()
            self.verify_metrics = self.hparams.verify_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            cer = self.cer_metrics.summarize("error_rate")
            clean_cer = self.clean_cer_metrics.summarize("error_rate")
            f1 = self.verify_metrics.summarize("F-score", threshold=0.5) * 100
            stage_stats = {"loss": stage_loss, "CER": cer, "F1": f1, "CleanCER": clean_cer}

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(cer)
            epoch_stats = {"epoch": epoch, "lr": old_lr}
            self.hparams.train_logger.log_stats(
                epoch_stats, {"loss": self.train_loss}, stage_stats
            )
            self.checkpointer.save_and_keep_only(
                meta={"CER": cer}, min_keys=["CER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.cer_file, 'w') as w:
                self.cer_metrics.write_stats(w)

if __name__ == "__main__":
    # This hack needed to import data preparation script from ../..
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
    from cslu_prepare import prepare_cslu  # noqa E402

    # Load hyperparameters file with command-line overrides
    hparams_file, overrides = sb.core.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = sb.yaml.load_extended_yaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Prepare data
    prepare_cslu(
        data_folder=hparams["data_folder"],
        save_folder=hparams["data_folder"],
    )

    train_set = hparams["train_loader"]()
    valid_set = hparams["valid_loader"]()
    hparams["ind2lab"] = hparams["train_loader"].label_dict["char"]["index2lab"]

    # Create brain object for training
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        checkpointer=hparams["checkpointer"],
        device=hparams["device"],
    )
    asr_brain.fit(hparams["epoch_counter"], train_set, valid_set)
    asr_brain.evaluate(hparams["test_loader"](), min_key="CER")
