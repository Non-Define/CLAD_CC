import os
import numpy as np

from calculate_modules import *
import a_dcf
import util

def calculate_minDCF_EER_CLLR_actDCF(cm_scores, cm_keys, output_file, printout=True):
    """
    Evaluation metrics for track 1
    Primary metrics: min DCF,
    Secondary metrics: EER, CLLR, actDCF
    """
    Pspoof = 0.05
    dcf_cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Cmiss': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa' : 10, # Cost of CM system falsely accepting nontarget speaker
    }

    assert cm_keys.size == cm_scores.size, "Error, unequal length of cm label and score files"
    
    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == util.g_cm_bon]
    spoof_cm = cm_scores[cm_keys == util.g_cm_spf]

    # EERs of the standalone systems
    eer_cm, frr, far, thresholds, eer_threshold = compute_eer(bona_cm, spoof_cm)#[0]
    # cllr
    cllr_cm = calculate_CLLR(bona_cm, spoof_cm)
    # min DCF
    minDCF_cm, _ = compute_mindcf(frr, far, thresholds, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])
    # actual DCF
    actDCF, _ = compute_actDCF(bona_cm, spoof_cm, Pspoof, dcf_cost_model['Cmiss'], dcf_cost_model['Cfa'])

    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tmin DCF \t\t= {} '
                        '(min DCF for countermeasure)\n'.format(
                            minDCF_cm))
            f_res.write('\tEER\t\t= {:8.9f} % '
                        '(EER for countermeasure)\n'.format(
                            eer_cm * 100))
            f_res.write('\tCLLR\t\t= {:8.9f} bits '
                        '(CLLR for countermeasure)\n'.format(
                            cllr_cm))
            f_res.write('\tactDCF\t\t= {:} '
                        '(actual DCF)\n'.format(
                            actDCF))
        os.system(f"cat {output_file}")

    return minDCF_cm, eer_cm, cllr_cm, actDCF