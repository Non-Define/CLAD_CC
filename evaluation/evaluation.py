#!/usr/bin/env python3
"""
main function to run evaluation package.

See usage in README.md
"""
import argparse
import sys

from calculate_metrics import calculate_minDCF_EER_CLLR_actDCF
from calculate_metrics import calculate_aDCF_tdcf_tEER
import util

def main(args: argparse.Namespace) -> None:
    
    # load score and keys
    cm_scores, cm_keys = util.load_cm_scores_keys(args.score_cm, args.key_cm)
    
    minDCF, eer, cllr, actDCF = calculate_minDCF_EER_CLLR_actDCF(
        cm_scores = cm_scores,
        cm_keys = cm_keys,
        output_file="./track1_result.txt")
    print("# Track 1 Result: \n")
    print("-eval_mindcf: {:.5f}\n-eval_eer (%): {:.3f}\n-eval_cllr (bits): {:.5f}\n-eval_actDCF: {:.5f}\n".format(
        minDCF, eer*100, cllr, actDCF))
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m",
                        dest="mode",
                        type=str,
                        help="mode flag: t1(Track 1) or t2_tandem(Track 2) or t2_single(Track 2)",
                        default="t1"
                        required=True)
    
    parser.add_argument("--cm",
                        dest="score_cm",
                        type=str,
                        help="cm score file as input")

    parser.add_argument("--cm_keys",
                        dest="key_cm",
                        type=str,
                        help="cm key file as input")

    main(parser.parse_args())