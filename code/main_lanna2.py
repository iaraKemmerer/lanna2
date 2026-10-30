"""
Trains a fixed CNN on one fold with one audio condition and evaluates it on the test children of that fold.
This produces the network results of the paper (Table "child-level AUC", the controls E2, E3b and E6 and the full
permutation test). Every call used in the paper is listed in run_all.sh.

Examples
--------
    python main_lanna2.py -a A0 -f 0 -c noise_for_all                  # noise for all, network A0 on fold 0 (5 runs)
    python main_lanna2.py -a A1 -f 1 -c original                       # original audio, network A1 on fold 1
    python main_lanna2.py -a A0 -f 0 -c noise_for_all --region silence # E2: the network only hears the silence
    python main_lanna2.py -a A0 -f 0 -c noise_for_all --truncate-ms 150   # E3b: every vowel cut to 150 ms
    python main_lanna2.py -a A0 -f 0 -c noise_for_all -v a             # E6: vowel /a/
    python main_lanna2.py -a A0 -f 0 -c noise_for_all --permute-labels 7  # one network of the full permutation test
"""
import argparse

from config_lanna2 import CFG
from experiment_runner_lanna2 import ExperimentRunner


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-a", "--arch", required=True, choices=["A0", "A1", "A2"], help="fixed network (architectures/<arch>.json)")
    ap.add_argument("-f", "--fold", type=int, required=True, choices=range(CFG.N_FOLDS), help="subject-disjoint fold")
    ap.add_argument("-c", "--condition", required=True, choices=CFG.CONDITIONS, help="audio condition for training, validation and test")
    ap.add_argument("-v", "--vowel", default="o", choices=list(CFG.VOWELS))
    ap.add_argument("-r", "--runs", type=int, default=None, help=f"runs (default {CFG.RUNS}; 1 with --permute-labels)")
    ap.add_argument("--also-test", nargs="*", default=["original"], choices=CFG.CONDITIONS,
                    help="other conditions on which the trained network is also tested (default: original)")
    ap.add_argument("--region", default="vowel", choices=["vowel", "silence"], help="silence: E2 control")
    ap.add_argument("--truncate-ms", type=float, default=None, help="cut every vowel to its first N ms (E3b control)")
    ap.add_argument("--snr-db", type=float, default=None, help=f"noise level of noise_for_all (default {CFG.NOISE_SNR_DB} dB below the vowel)")
    ap.add_argument("--augmentation", type=int, default=CFG.AUGMENTATION_FACTOR, help="copies per majority-class child (0 = none)")
    ap.add_argument("--epochs", type=int, default=CFG.MAX_EPOCHS)
    ap.add_argument("--permute-labels", type=int, default=None, help="seed: shuffle the training labels among children")
    ap.add_argument("--device", default=CFG.DEVICE)
    ap.add_argument("--name", default=None, help="output folder name (default: built from the options)")
    args = ap.parse_args()
    CFG.MAX_EPOCHS = args.epochs
    ExperimentRunner(args.arch, args.fold, args.condition, args.vowel, args.region, args.truncate_ms, args.snr_db,
                     args.also_test, args.augmentation, args.runs, args.permute_labels, args.device, args.name).run_experiment()


if __name__ == "__main__":
    main()
