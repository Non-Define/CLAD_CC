import os
import numpy as np

from evaluation.calculate_modules import *
import evaluation.util

def calculate_minDCF_EER_CLLR(cm_scores_file,
                       output_file,
                       printout=True):
    # Evaluation metrics for Phase 1
    # Primary metrics: min DCF,
    # Secondary metrics: EER, CLLR

    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Cmiss': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa' : 10, # Cost of CM system falsely accepting nontarget speaker
    }


    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_keys = cm_data[:, 3]
    cm_scores = cm_data[:, 2].astype(np.float64)

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_cm, frr, far, thresholds = compute_eer(bona_cm, spoof_cm)#[0]
    cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tmin DCF \t\t= {} % '
                        '(min DCF for countermeasure)\n'.format(
                            minDCF_cm))
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(EER for countermeasure)\n'.format(
                            eer_cm * 100))
            f_res.write('\tCLLR\t\t= {:8.9f} % '
                        '(CLLR for countermeasure)\n'.format(
                            cllr_cm * 100))
        os.system(f"cat {output_file}")

    return minDCF_cm, eer_cm, cllr_cm