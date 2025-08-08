import sys
import numpy as np

def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):
    """
    compute ASV error rates
    
    input
    -----
      tar_asv: np.array, (#N, ), target bonafide scores
      non_asv: np.array, (#M, ), nontarget bonafide scores
      spoof_asv: np.array, (#K, ), spoof scores
      asv_threshold: scalar, ASV threshold

    output
    ------
      Pfa_asv: scalar, false acceptance rate of nontarget bonafide
      Pmiss_asv: scalar, miss rate of target bonafide
      Pmiss_spoof_asv: scalar, 1 - Pfa_spoof_asv
      Pfa_spoof_asv: scalar, false acceptance rate of spoofed data
    """
    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):
    """
    compute DET curve values
                                                                           
    input
    -----
      target_scores:    np.array, target trial scores
      nontarget_scores: np.array, nontarget trial scores
    
    output
    ------
      frr:   np.array, FRR, (#N, ), where #N is total number of scores + 1
      far:   np.array, FAR, (#N, ), where #N is total number of scores + 1
      thr:   np.array, threshold, (#N, )
    """

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - \
        (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate(
        (np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                          nontarget_scores.size))  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_Pmiss_Pfa_Pspoof_curves(tar_scores, non_scores, spf_scores):

    # Concatenate all scores and designate arbitrary labels 1=target, 0=nontarget, -1=spoof
    all_scores = np.concatenate((tar_scores, non_scores, spf_scores))
    labels = np.concatenate((np.ones(tar_scores.size), np.zeros(non_scores.size), -1*np.ones(spf_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Cumulative sums
    tar_sums    = np.cumsum(labels==1)
    non_sums    = np.cumsum(labels==0)
    spoof_sums  = np.cumsum(labels==-1)

    Pmiss       = np.concatenate((np.atleast_1d(0), tar_sums / tar_scores.size))
    Pfa_non     = np.concatenate((np.atleast_1d(1), 1 - (non_sums / non_scores.size)))
    Pfa_spoof   = np.concatenate((np.atleast_1d(1), 1 - (spoof_sums / spf_scores.size)))
    thresholds  = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return Pmiss, Pfa_non, Pfa_spoof, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, frr, far, thresholds, thresholds[min_index]


def compute_mindcf(frr, far, thresholds, Pspoof, Cmiss, Cfa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds

    p_target = 1- Pspoof
    for i in range(0, len(frr)):
        # Weighted sum of false negative and false positive errors.
        c_det = Cmiss * frr[i] * p_target + Cfa * far[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(Cmiss * p_target, Cfa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def compute_actDCF(bonafide_scores, spoof_scores, Pspoof, Cmiss, Cfa):
    """
    compute actual DCF, given threshold decided by prior and decision costs

    input
    -----
      bonafide_scores: np.array, scores of bonafide data
      spoof_scores: np.array, scores of spoof data
      Pspoof: scalar, prior probabiltiy of spoofed class
      Cmiss: scalar, decision cost of missing a bonafide sample
      Cfa: scalar, decision cost of falsely accept a spoofed sample

    output
    ------
      actDCF: scalar, actual DCF normalized
      threshold: scalar, threshold for making the decision
    """
    # the beta in evaluation plan (eq.(3))
    beta = Cmiss * (1 - Pspoof) / (Cfa * Pspoof)
    
    # compute the decision threshold based on
    threshold = - np.log(beta)

    # miss rate
    rate_miss = np.sum(bonafide_scores < threshold) / bonafide_scores.size

    # fa rate
    rate_fa = np.sum(spoof_scores >= threshold) / spoof_scores.size

    # unnormalized DCF
    act_dcf = Cmiss * (1 - Pspoof) * rate_miss + Cfa * Pspoof * rate_fa

    # normalized DCF
    act_dcf = act_dcf / np.min([Cfa * Pspoof, Cmiss * (1 - Pspoof)])
    
    return act_dcf, threshold
    

def calculate_CLLR(target_llrs, nontarget_llrs):
    """
    Calculate the CLLR of the scores.
    
    Parameters:
    target_llrs (list or numpy array): Log-likelihood ratios for target trials.
    nontarget_llrs (list or numpy array): Log-likelihood ratios for non-target trials.
    
    Returns:
    float: The calculated CLLR value.
    """
    def negative_log_sigmoid(lodds):
        """
        Calculate the negative log of the sigmoid function.
        
        Parameters:
        lodds (numpy array): Log-odds values.
        
        Returns:
        numpy array: The negative log of the sigmoid values.
        """
        return np.log1p(np.exp(-lodds))

    # Convert the input lists to numpy arrays if they are not already
    target_llrs = np.array(target_llrs)
    nontarget_llrs = np.array(nontarget_llrs)
    
    # Calculate the CLLR value
    cllr = 0.5 * (np.mean(negative_log_sigmoid(target_llrs)) + np.mean(negative_log_sigmoid(-nontarget_llrs))) / np.log(2)
    
    return cllr