import numpy as np
import copy
from astropy.io import fits
from astropy.table import Table
from fast3tree import fast3tree
from utils import cosmo, get_comoving_distance, get_comoving_distance_h

import Corrfunc
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp


# Comoving pi_max and wp bins in Mpc/h
pi_max = 40
rp_bins_wp = np.logspace(-0.8, 1.6, 13)


def wprp(gal, random, pi_max = pi_max, rp_bins = rp_bins_wp,
         gal_weights = 1.0, random_weights = 1.0, nthread = 55):
    DD = DDrppi_mocks(True, 3, nthread, pi_max, rp_bins,
                      copy.deepcopy(gal[:, 0]), copy.deepcopy(gal[:, 1]),
                      get_comoving_distance_h(copy.deepcopy(gal[:, 2])),
                      weights1 = gal_weights,
                      is_comoving_dist = True,
                      weight_type = 'pair_product')['npairs']
    DR = DDrppi_mocks(False, 3, nthread, pi_max, rp_bins,
                      copy.deepcopy(gal[:, 0]), copy.deepcopy(gal[:, 1]),
                      get_comoving_distance_h(copy.deepcopy(gal[:, 2])),
                      weights1 = gal_weights,
                      RA2 = copy.deepcopy(random[:, 0]),
                      DEC2 = copy.deepcopy(random[:, 1]),
                      CZ2 = get_comoving_distance_h(
                            copy.deepcopy(random[:, 2])),
                      weights2 = random_weights,
                      is_comoving_dist = True,
                      weight_type = 'pair_product')['npairs']
    RR = DDrppi_mocks(True, 3, nthread, pi_max, rp_bins,
                      copy.deepcopy(random[:, 0]), copy.deepcopy(random[:, 1]),
                      get_comoving_distance_h(
                            copy.deepcopy(random[:, 2])),
                      weights1 = random_weights,
                      is_comoving_dist = True,
                      weight_type = 'pair_product')['npairs']
    wp = convert_rp_pi_counts_to_wp(len(gal), len(gal),
                                    len(random), len(random),
                                    DD, DR, DR, RR,
                                    len(rp_bins) - 1, pi_max)
    return wp

def weighted_wprp(gal, gal_weights, random, pi_max, rp_bins,
                 nthread = 55):
    # random_weights = 1.0,
    gal_weights = np.array(gal_weights)
    random_weights = copy.deepcopy(gal_weights)
    # len(random) > len(gal_weights)
    additional_elements = np.random.choice(gal_weights, size=len(random) - len(gal_weights), replace=True)
    extended_random_weights = np.concatenate([random_weights, additional_elements])

    np.random.shuffle(extended_random_weights)

    DD = DDrppi_mocks(True, 3, nthread, pi_max, rp_bins,
                      copy.deepcopy(gal[:, 0]), copy.deepcopy(gal[:, 1]),
                      get_comoving_distance_h(copy.deepcopy(gal[:, 2])),
                      weights1 = gal_weights,
                      is_comoving_dist = True,
                      weight_type = 'pair_product')
    DD_npairs   = DD['npairs']
    DD_weight   = DD['weightavg']
    DD_weighted = DD_npairs*DD_weight

    DR = DDrppi_mocks(False, 3, nthread, pi_max, rp_bins,
                      copy.deepcopy(gal[:, 0]), copy.deepcopy(gal[:, 1]),
                      get_comoving_distance_h(copy.deepcopy(gal[:, 2])),
                      weights1 = gal_weights,
                      RA2 = copy.deepcopy(random[:, 0]),
                      DEC2 = copy.deepcopy(random[:, 1]),
                      CZ2 = get_comoving_distance_h(
                            copy.deepcopy(random[:, 2])),
                      weights2 = extended_random_weights,
                      is_comoving_dist = True,
                      weight_type = 'pair_product')
    DR_npairs   = DR['npairs']
    DR_weight   = DR['weightavg']
    DR_weighted = DR_npairs*DR_weight

    RR = DDrppi_mocks(True, 3, nthread, pi_max, rp_bins,
                      copy.deepcopy(random[:, 0]), copy.deepcopy(random[:, 1]),
                      get_comoving_distance_h(
                            copy.deepcopy(random[:, 2])),
                      weights1 = extended_random_weights,
                      is_comoving_dist = True,
                      weight_type = 'pair_product')
    RR_npairs   = RR['npairs']
    RR_weight   = RR['weightavg']
    RR_weighted = RR_npairs*RR_weight

    # wp = convert_rp_pi_counts_to_wp(len(gal), len(gal),
    #                                 len(random), len(random),
    #                                 DD, DR, DR, RR,
    #                                 len(rp_bins) - 1, pi_max)
    weighted_wp = convert_rp_pi_counts_to_wp(len(gal), len(gal),
                                    len(random), len(random),
                                    DD_weighted, DR_weighted, DR_weighted, RR_weighted,
                                    len(rp_bins) - 1, pi_max)
    return weighted_wp

def weighted_wprp_theory(gal, gal_weights, random, pi_max, rp_bins, boxsize,
                        nthread = 55):
    # random_weights = 1.0,
    gal_weights = np.array(gal_weights)
    random_weights = copy.deepcopy(gal_weights)
    # len(random) > len(gal_weights)
    additional_elements = np.random.choice(gal_weights, size=len(random) - len(gal_weights), replace=True)
    extended_random_weights = np.concatenate([random_weights, additional_elements])

    np.random.shuffle(extended_random_weights)

    DD = Corrfunc.theory.DDrppi(True, nthread, pi_max, rp_bins,
                                copy.deepcopy(gal[:, 0]), copy.deepcopy(gal[:, 1]), copy.deepcopy(gal[:, 2]),
                                weights1 = gal_weights,
                                boxsize  = boxsize,
                                weight_type = 'pair_product')
    DD_npairs   = DD['npairs']
    DD_weight   = DD['weightavg']
    DD_weighted = DD_npairs*DD_weight

    DR = Corrfunc.theory.DDrppi(False, nthread, pi_max, rp_bins,
                                copy.deepcopy(gal[:, 0]), copy.deepcopy(gal[:, 1]), copy.deepcopy(gal[:, 2]),
                                weights1 = gal_weights,
                                boxsize  = boxsize,
                                X2 = copy.deepcopy(random[:, 0]),
                                Y2 = copy.deepcopy(random[:, 1]),
                                Z2 = copy.deepcopy(random[:, 2]),
                                weights2 = extended_random_weights,
                                weight_type = 'pair_product')
    DR_npairs   = DR['npairs']
    DR_weight   = DR['weightavg']
    DR_weighted = DR_npairs*DR_weight

    RR = Corrfunc.theory.DDrppi(True, nthread, pi_max, rp_bins,
                                copy.deepcopy(random[:, 0]), copy.deepcopy(random[:, 1]), copy.deepcopy(random[:, 2]),
                                weights1 = extended_random_weights,
                                boxsize  = boxsize,
                                weight_type = 'pair_product')
    RR_npairs   = RR['npairs']
    RR_weight   = RR['weightavg']
    RR_weighted = RR_npairs*RR_weight

    # wp = convert_rp_pi_counts_to_wp(len(gal), len(gal),
    #                                 len(random), len(random),
    #                                 DD, DR, DR, RR,
    #                                 len(rp_bins) - 1, pi_max)
    weighted_wp = convert_rp_pi_counts_to_wp(len(gal), len(gal),
                                    len(random), len(random),
                                    DD_weighted, DR_weighted, DR_weighted, RR_weighted,
                                    len(rp_bins) - 1, pi_max)
    return weighted_wp


# Comoving cylinder size in Mpc/h, cic bins
r_out, l_cyl = 2, 10
cic_bin = np.concatenate([np.arange(10),
                          np.around(np.logspace(
                          1, np.log10(150), 30 - 10)).astype(int)]) - 0.5


def Ncic(centers, companions, r_cyl, l_cyl = l_cyl,
         weigh_companions = False, companion_weights = None, 
         weigh_counts = False, count_weights = None):
    
    centers_ = copy.deepcopy(centers)
    centers_[:, 0] = np.radians(centers_[:, 0])
    centers_[:, 1] = np.radians(centers_[:, 1])
    centers_[:, 2] = get_comoving_distance(centers_[:, 2])

    companions_ = copy.deepcopy(companions)
    companions_[:, 0] = np.radians(companions_[:, 0])
    companions_[:, 1] = np.radians(companions_[:, 1])
    companions_[:, 2] = get_comoving_distance(companions_[:, 2])

    with fast3tree(companions_[:, :2]) as tree_xy:

        cnts = []
        for pt in centers_:
            ang_radius = r_cyl / cosmo.h / (pt[2] - l_cyl) #rad
            idx_xy = tree_xy.query_radius(pt[:2], ang_radius / np.cos(pt[1]))
            cnt = 0
            for j in idx_xy:
                if (np.abs(pt[2] - companions_[j, 2]) < l_cyl / cosmo.h and
                ((pt[0] - companions_[j, 0]) * np.cos(pt[1])) ** 2 +
                ((pt[1] - companions_[j, 1])) ** 2 < (r_cyl / cosmo.h / companions_[j, 2]) ** 2
                and not np.allclose(pt, companions_[j], 0, 1e-7)):
                    # count does not include center itself
                    if weigh_companions:
                        cnt += companion_weights[j]
                    else:
                        cnt += 1
            cnts.append(cnt)
    
    if weigh_counts:
        cnts = np.array(cnts) * count_weights
    else:
        cnts = np.array(cnts)
            
    return cnts


# def PNc(centers, companions, r_out = r_out,
#         l_cyl = l_cyl, cic_bin = cic_bin,
#         weigh_centers = False, center_weights = None, 
#         weigh_companions = False, companion_weights = None, 
#         weigh_counts = False, count_weights = None):
#     """
#     center_weights is 1/fs_center
#     companion_weights is 1/fs_companion
#     count_weights is 1/AC_center
#     """
    
#     Nc_out = Ncic(centers, companions, r_out, l_cyl,
#                   weigh_companions = weigh_companions, companion_weights = companion_weights, 
#                   weigh_counts = weigh_counts, count_weights = count_weights)
#     if weigh_centers:
#         Pcic = np.histogram(Nc_out, bins = cic_bin, weights = center_weights)[0] / np.sum(center_weights)
#     else:
#         Pcic = np.histogram(Nc_out, bins = cic_bin)[0] / len(centers)
    
#     return Pcic

def dPdN(centers, companions, r_out = r_out,
        l_cyl = l_cyl, cic_bin = cic_bin,
        weigh_centers = False, center_weights = None, 
        weigh_companions = False, companion_weights = None, 
        weigh_counts = False, count_weights = None):
    """
    center_weights is 1/fs_center
    companion_weights is 1/fs_companion
    count_weights is 1/AC_center
    """
    
    Nc_out = Ncic(centers, companions, r_out, l_cyl,
                  weigh_companions = weigh_companions, companion_weights = companion_weights, 
                  weigh_counts = weigh_counts, count_weights = count_weights)
    if weigh_centers:
        Pcic = np.histogram(Nc_out, bins = cic_bin, weights = center_weights)[0] / np.sum(center_weights)
        dPdN = Pcic / np.diff(cic_bin)
    else:
        # Pcic = np.histogram(Nc_out, bins = cic_bin)[0] / len(centers)
        # dPdN = Pcic / np.diff(cic_bin)
        dPdN = np.histogram(Nc_out, bins=cic_bin, density=True)[0]
    
    # return Pcic
    return dPdN

def dPdN_Nc_out(centers, companions, r_out = r_out,
        l_cyl = l_cyl, 
        weigh_centers = False, center_weights = None, 
        weigh_companions = False, companion_weights = None, 
        weigh_counts = False, count_weights = None):
    """
    center_weights is 1/fs_center
    companion_weights is 1/fs_companion
    count_weights is 1/AC_center
    """
    
    Nc_out = Ncic(centers, companions, r_out, l_cyl,
                  weigh_companions = weigh_companions, companion_weights = companion_weights, 
                  weigh_counts = weigh_counts, count_weights = count_weights)
   
    return Nc_out
