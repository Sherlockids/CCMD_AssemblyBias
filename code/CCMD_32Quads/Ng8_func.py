import numpy as np
from astropy.io import fits
import copy
import collections
from astropy.cosmology import Planck13  # Planck (2014) Paper XVI, Table 5
from halotools.mock_observables import apply_zspace_distortion

import sys
sys.path.append('..') 

from utils import Vcom, get_comoving_distance_h, c, z_min
# from data_stat_func import wprp, dPdN
from data_stat_func import weighted_wprp
from data_stat_func import wprp

import Corrfunc
# from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
# from Corrfunc.utils import convert_rp_pi_counts_to_wp

###########################################################
# Corrfunc import failed
import sys
sys.path.append('..') 
from Corrfunc.mocks import DDrppi_mocks #2026/2/14 wenlk
from Corrfunc_utils import convert_rp_pi_counts_to_wp #2026/2/14 wenlk
###########################################################


from scipy.spatial import KDTree

def ra_dec_z_to_cartesian(ra, dec, z_redshift):
    """turn (RA, Dec, z_redshift) into (x, y, z)[Mpc/h]"""
    cosmo = Planck13
    comoving_dist = cosmo.comoving_distance(z_redshift).value * cosmo.h  # d[Mpc] => d'[Mpc/h].  
    # d'[Mpc/h] / h = d[Mpc]  => d'[Mpc/h] = d[Mpc] * cosmo.h
    
    ra_rad  = np.radians(ra) #RA, in radians
    dec_rad = np.radians(dec)#DEC, in radians
    
    x = comoving_dist * np.cos(dec_rad) * np.cos(ra_rad)
    y = comoving_dist * np.cos(dec_rad) * np.sin(ra_rad)
    z = comoving_dist * np.sin(dec_rad)
    
    return np.column_stack((x, y, z))

def N_g_8(coords_RA, coords_DEC, coords_z, ):
    '''
    coords_RA/DEC/z: array like
    '''
    coords = ra_dec_z_to_cartesian(coords_RA, coords_DEC, coords_z)

    tree   = KDTree(coords)
    radius = 8.0  # 8 Mpc/h
    neighbor_counts_masked = tree.query_ball_point(coords, radius, return_length=True)
    neighbor_counts_masked = neighbor_counts_masked - 1

    return neighbor_counts_masked


def N_g_8_cross(A_coords, B_coords, radius = 8):
    """
    Calculate the number of B_coords points around A_coords.

    Parameters:
    A_coords: (n, 3), [A_ra, A_dec, A_redshift]
    B_coords: (m, 3), [B_ra, B_dec, B_redshift], volume limited sample
    radius: radius of the ball, default 8 Mpc/h
    
    Return:
    counts: array like, (n, 1)
    """

    A_coords_xyz = ra_dec_z_to_cartesian(A_coords[:, 0], A_coords[:, 1], A_coords[:, 2])
    B_coords_xyz = ra_dec_z_to_cartesian(B_coords[:, 0], B_coords[:, 1], B_coords[:, 2])

    # KDTree of B_coords
    tree = KDTree(B_coords_xyz)
    
    # Count neightbours
    neighbor_counts_masked = tree.query_ball_point(A_coords_xyz, radius, return_length=True) - 1

    return neighbor_counts_masked
    

def N_g_8_theory(   coords_X, coords_Y, coords_Z, box_size = 1000): 
    '''
    coords_X/Y/Z: array like
    data_Mr: array like, len(data_Mr) = len(coords)
    data_gmr: array like, len(data_gmr) = len(coords)
    Mr_min/max: float, threshold of Mr
    gmr_min/max: float, threshold of gmr
    '''
    coords = np.array([coords_X, coords_Y, coords_Z]).T
    tree   = KDTree(coords, boxsize = box_size)
    radius = 8.0  # 8 Mpc/h
    neighbor_counts_masked = tree.query_ball_point(coords, radius, return_length=True)
    neighbor_counts_masked = neighbor_counts_masked - 1

    return neighbor_counts_masked

def N_g_8_theory_cross(A_coords, B_coords, box_size = 1000, radius = 8.0): 
    """
    Calculate the number of B_coords points around A_coords.

    Parameters:
    # A_coords: (n, 3), [A_x, A_y, A_z]
    B_coords: (m, 3), [B_x, B_y, B_z], volume limited sample
    box_size: the box size, default 1000 Mpc/h
    radius: radius of the ball, default 8 Mpc/h
    
    Return:
    counts: array like, (n, 1)
    """
    tree   = KDTree(B_coords, boxsize = box_size)
    neighbor_counts_masked = tree.query_ball_point(A_coords, radius, return_length=True) - 1

    return neighbor_counts_masked

def DD_rp_pi_to_DD_rp(DD_rp_pi, rp_bins):
    len_rp = len(rp_bins) - 1
    len_pi = len(DD_rp_pi) // len_rp
    DD_rp  = DD_rp_pi.reshape(len_rp, len_pi).sum(axis = 1)
    return DD_rp

def cross_correlation_wprp(   RA1, DEC1, CZ1, rand_RA1, rand_DEC1, rand_CZ1, 
                              RA2, DEC2, CZ2, rand_RA2, rand_DEC2, rand_CZ2,
                              pi_max, rp_bins, gal_weights = 1.0, random_weights = 1.0, nthread = 55):
      '''
      calculate the cross correlation of D1, D2, R1, R2 using Landy-Szalay estimator
      '''
      #  estimator == "Landy-Szalay":
      ND1 = len(RA1)
      ND2 = len(RA2)
      NR1 = len(rand_RA1)
      NR2 = len(rand_RA2)

      factor1 = ND1 * ND2 / (NR1 * NR2)
      factor2 = ND1 * NR2 / (NR1 * NR2)
      factor3 = ND2 * NR1 / (NR1 * NR2)

      # (D1D2 - D1R2 - D2R1 + R1R2)/R1R2
      D1D2 = DDrppi_mocks(False, 3, nthread, pi_max, rp_bins,
                              copy.deepcopy(RA1), copy.deepcopy(DEC1),
                              get_comoving_distance_h(copy.deepcopy(CZ1)),
                              weights1 = gal_weights,
                              RA2      = copy.deepcopy(RA2),
                              DEC2     = copy.deepcopy(DEC2),
                              CZ2      = get_comoving_distance_h(copy.deepcopy(CZ2)),
                              weights2 = random_weights,
                              is_comoving_dist = True,
                              weight_type = 'pair_product')['npairs']
      
      D1R2 = DDrppi_mocks(False, 3, nthread, pi_max, rp_bins,
                              copy.deepcopy(RA1), copy.deepcopy(DEC1),
                              get_comoving_distance_h(copy.deepcopy(CZ1)),
                              weights1 = gal_weights,
                              RA2      = copy.deepcopy(rand_RA2),
                              DEC2     = copy.deepcopy(rand_DEC2),
                              CZ2      = get_comoving_distance_h(copy.deepcopy(rand_CZ2)),
                              weights2 = random_weights,
                              is_comoving_dist = True,
                              weight_type = 'pair_product')['npairs']
      
      D2R1 = DDrppi_mocks(False, 3, nthread, pi_max, rp_bins,
                              copy.deepcopy(RA2), copy.deepcopy(DEC2),
                              get_comoving_distance_h(copy.deepcopy(CZ2)),
                              weights1 = gal_weights,
                              RA2      = copy.deepcopy(rand_RA1),
                              DEC2     = copy.deepcopy(rand_DEC1),
                              CZ2      = get_comoving_distance_h(copy.deepcopy(rand_CZ1)),
                              weights2 = random_weights,
                              is_comoving_dist = True,
                              weight_type = 'pair_product')['npairs']
      
      R1R2 = DDrppi_mocks(False, 3, nthread, pi_max, rp_bins,
                              copy.deepcopy(rand_RA1), copy.deepcopy(rand_DEC1),
                              get_comoving_distance_h(copy.deepcopy(rand_CZ1)),
                              weights1 = gal_weights,
                              RA2      = copy.deepcopy(rand_RA2),
                              DEC2     = copy.deepcopy(rand_DEC2),
                              CZ2      = get_comoving_distance_h(copy.deepcopy(rand_CZ2)),
                              weights2 = random_weights,
                              is_comoving_dist = True,
                              weight_type = 'pair_product')['npairs']
    
      term1 = (1.0 / factor1) * (D1D2 / R1R2)
      term2 = (1.0 / factor2) * (D1R2 / R1R2)
      term3 = (1.0 / factor3) * (D2R1 / R1R2)
      wp_rp_pi = term1 - term2 - term3 + 1.0
      wp_rp    = DD_rp_pi_to_DD_rp(wp_rp_pi, rp_bins)
    
      return wp_rp

def cross_correlation_wprp_theory( X1, Y1, Z1, rand_X1, rand_Y1, rand_Z1, 
                                    X2, Y2, Z2, rand_X2, rand_Y2, rand_Z2,
                                    pi_max, rp_bins, boxsize, gal_weights = 1.0, random_weights = 1.0, nthread = 55):
      '''
      calculate the cross correlation of D1, D2, R1, R2 using Landy-Szalay estimator
      '''
      #  estimator == "Landy-Szalay":
      ND1 = len(X1)
      ND2 = len(X2)
      NR1 = len(rand_X1)
      NR2 = len(rand_X2)

      factor1 = ND1 * ND2 / (NR1 * NR2)
      factor2 = ND1 * NR2 / (NR1 * NR2)
      factor3 = ND2 * NR1 / (NR1 * NR2)

      # (D1D2 - D1R2 - D2R1 + R1R2)/R1R2
      D1D2 = Corrfunc.theory.DDrppi(False, nthread, pi_max, rp_bins,
                                    copy.deepcopy(X1), copy.deepcopy(Y1), copy.deepcopy(Z1),
                                    weights1 = gal_weights,
                                    boxsize  = boxsize,
                                    X2       = copy.deepcopy(X2),
                                    Y2       = copy.deepcopy(Y2),
                                    Z2       = copy.deepcopy(Z2),
                                    weights2 = random_weights,
                                    weight_type = 'pair_product')['npairs']
      
      D1R2 = Corrfunc.theory.DDrppi(False, nthread, pi_max, rp_bins,
                                    copy.deepcopy(X1), copy.deepcopy(Y1), copy.deepcopy(Z1),
                                    weights1 = gal_weights,
                                    boxsize  = boxsize,
                                    X2       = copy.deepcopy(rand_X2),
                                    Y2       = copy.deepcopy(rand_Y2),
                                    Z2       = copy.deepcopy(rand_Z2),
                                    weights2 = random_weights,
                                    weight_type = 'pair_product')['npairs']
      
      D2R1 = Corrfunc.theory.DDrppi(False, nthread, pi_max, rp_bins,
                                    copy.deepcopy(X2), copy.deepcopy(Y2), copy.deepcopy(Z2),
                                    weights1 = gal_weights,
                                    boxsize  = boxsize,
                                    X2       = copy.deepcopy(rand_X1),
                                    Y2       = copy.deepcopy(rand_Y1),
                                    Z2       = copy.deepcopy(rand_Z1),
                                    weights2 = random_weights,
                                    weight_type = 'pair_product')['npairs']
      
      R1R2 = Corrfunc.theory.DDrppi(False, nthread, pi_max, rp_bins,
                                    copy.deepcopy(rand_X1), copy.deepcopy(rand_Y1), copy.deepcopy(rand_Z1),
                                    weights1 = gal_weights,
                                    boxsize  = boxsize,
                                    X2       = copy.deepcopy(rand_X2),
                                    Y2       = copy.deepcopy(rand_Y2),
                                    Z2       = copy.deepcopy(rand_Z2),
                                    weights2 = random_weights,
                                    weight_type = 'pair_product')['npairs']
    
      term1 = (1.0 / factor1) * (D1D2 / R1R2)
      term2 = (1.0 / factor2) * (D1R2 / R1R2)
      term3 = (1.0 / factor3) * (D2R1 / R1R2)
      wp_rp_pi = term1 - term2 - term3 + 1.0
      wp_rp    = DD_rp_pi_to_DD_rp(wp_rp_pi, rp_bins)
    
      return wp_rp

def rand_z_assign(z1, z2):
    '''
    z1: array like
    z2: array like, len(z2) < len(z1)

    sample randz from z2, and make len(z2) = len(z1)
    '''
    randz = np.tile(z2, len(z1) // len(z2) + 1 )[:len(z1)]

    return randz

def select_bottom20_in_random(x, weight, percentile=20):
    """
    根据权重筛选前 percentile% 的元素，
    若相同权重的元素超过比例限制，则随机抽取。
    
    参数:
        x: 数据数组 (np.array 或 list)
        weight: 权重数组 (np.array 或 list)
        percentile: 百分比（默认20）
    
    返回:
        筛选后的 x 的子集
    """
    x = np.array(x)
    weight = np.array(weight)
    
    # 计算分界值（前20%的权重阈值）
    threshold = np.percentile(weight, percentile)
    
    # 1. 直接取出权重严格小于阈值的元素
    mask_strictly_below = weight < threshold
    selected_below = x[mask_strictly_below]
    
    # 2. 处理权重等于阈值的元素
    mask_equal = weight == threshold
    x_equal = x[mask_equal]
    
    # 计算还需抽取的数量
    total_desired = int(len(x) * percentile / 100)  # 期望的总数量
    remaining_needed = total_desired - len(selected_below)
    
    # 随机抽取剩余需要的元素
    if remaining_needed > 0 and len(x_equal) > 0:
        np.random.seed(42)  # 固定随机种子（可选）
        selected_equal = np.random.choice(
            x_equal, 
            size=min(remaining_needed, len(x_equal)), 
            replace=False
        )
        result = np.concatenate([selected_below, selected_equal])
    else:
        result = selected_below
    
    return result

def select_top20_in_random(x, weight, percentile=20):
    """
    根据权重筛选前 percentile% 的元素，
    若相同权重的元素超过比例限制，则随机抽取。
    
    参数:
        x: 数据数组 (np.array 或 list)
        weight: 权重数组 (np.array 或 list)
        percentile: 百分比（默认20）
    
    返回:
        筛选后的 x 的子集
    """
    x = np.array(x)
    weight = np.array(weight)
    
    # top20%, in percentile is 80%
    threshold = np.percentile(weight, 100-percentile)
    
    # 1. 直接取出权重严格大于阈值的元素
    mask_strictly_above = weight > threshold
    selected_above = x[mask_strictly_above]
    
    # 2. 处理权重等于阈值的元素
    mask_equal = weight == threshold
    x_equal = x[mask_equal]
    
    # 计算还需抽取的数量
    total_desired = int(len(x) * percentile / 100)  # 期望的总数量
    remaining_needed = total_desired - len(selected_above)
    
    # 随机抽取剩余需要的元素
    if remaining_needed > 0 and len(x_equal) > 0:
        np.random.seed(42)  # 固定随机种子（可选）
        selected_equal = np.random.choice(
            x_equal, 
            size=min(remaining_needed, len(x_equal)), 
            replace=False
        )
        result = np.concatenate([selected_above, selected_equal])
    else:
        result = selected_above
    
    return result

def Ng8_percentile_wprp(gal, random, pi_max, rp_bins, nthread = 55):
                        # z_min, z_max, 
                        # data_Mr , Mr_min , Mr_max, 
                        # data_gmr, gmr_min, gmr_max,):
    
    D1_ra  = copy.deepcopy(gal[:, 0])
    D1_dec = copy.deepcopy(gal[:, 1])
    D1_z   = copy.deepcopy(gal[:, 2])
    R1_ra  = copy.deepcopy(random[:, 0])
    R1_dec = copy.deepcopy(random[:, 1])
    R1_z   = copy.deepcopy(random[:, 2])


    Ng8_thr = N_g_8(    D1_ra, D1_dec, D1_z, )
 
    # len(Ng8_thr) = len(D1_ra)
    ######################################################################################################################
    # cross correlation with data and data(Ng8 bottom 20%)

    D2_ra  = select_bottom20_in_random(D1_ra , Ng8_thr)
    D2_dec = select_bottom20_in_random(D1_dec, Ng8_thr)
    D2_z   = select_bottom20_in_random(D1_z  , Ng8_thr)

    R2_ra  = R1_ra
    R2_dec = R1_dec
    R2_z   = rand_z_assign(R1_z, D2_z) # shuffle z from D2_z(thr masked)

    wprp_bottom20 = cross_correlation_wprp( D1_ra, D1_dec, D1_z, R1_ra, R1_dec, R1_z, 
                                            D2_ra, D2_dec, D2_z, R2_ra, R2_dec, R2_z,
                                            pi_max, rp_bins, gal_weights = 1.0, random_weights = 1.0, nthread = nthread)
    ######################################################################################################################
    # cross correlation with data and data(Ng8 top 20%)

    D2_ra  = select_top20_in_random(D1_ra , Ng8_thr)
    D2_dec = select_top20_in_random(D1_dec, Ng8_thr)
    D2_z   = select_top20_in_random(D1_z  , Ng8_thr)

    R2_ra  = R1_ra
    R2_dec = R1_dec
    R2_z   = rand_z_assign(R1_z, D2_z) # shuffle z from D2_z(thr masked)

    wprp_top20 = cross_correlation_wprp( D1_ra, D1_dec, D1_z, R1_ra, R1_dec, R1_z, 
                                            D2_ra, D2_dec, D2_z, R2_ra, R2_dec, R2_z,
                                            pi_max, rp_bins, gal_weights = 1.0, random_weights = 1.0, nthread = nthread)
    ######################################################################################################################
    return wprp_bottom20, wprp_top20

def Ng8_percentile_wprp_theory(gal, random, pi_max, rp_bins, boxsize, nthread = 55):
    D1_X = copy.deepcopy(gal[:, 0])
    D1_Y = copy.deepcopy(gal[:, 1])
    D1_Z = copy.deepcopy(gal[:, 2])
    R1_X = copy.deepcopy(random[:, 0])
    R1_Y = copy.deepcopy(random[:, 1])
    R1_Z = copy.deepcopy(random[:, 2])

    Ng8_thr = N_g_8_theory(    D1_X, D1_Y, D1_Z, )
 
    # len(Ng8_thr) = len(D1_ra)
    ######################################################################################################################
    # cross correlation with data and data(Ng8 bottom 20%)

    D2_X = select_bottom20_in_random(D1_X, Ng8_thr)
    D2_Y = select_bottom20_in_random(D1_Y, Ng8_thr)
    D2_Z = select_bottom20_in_random(D1_Z, Ng8_thr)

    R2_X = R1_X
    R2_Y = R1_Y
    R2_Z = R1_Z

    wprp_bottom20 = cross_correlation_wprp_theory(  D1_X, D1_Y, D1_Z, R1_X, R1_Y, R1_Z, 
                                                    D2_X, D2_Y, D2_Z, R2_X, R2_Y, R2_Z,
                                                    pi_max, rp_bins, boxsize, gal_weights  = 1.0, 
                                                    random_weights = 1.0, nthread = nthread)
    ######################################################################################################################
    # cross correlation with data and data(Ng8 top 20%)

    D2_X = select_top20_in_random(D1_X, Ng8_thr)
    D2_Y = select_top20_in_random(D1_Y, Ng8_thr)
    D2_Z = select_top20_in_random(D1_Z, Ng8_thr)

    R2_X = R1_X
    R2_Y = R1_Y
    R2_Z = R1_Z

    wprp_top20 = cross_correlation_wprp_theory( D1_X, D1_Y, D1_Z, R1_X, R1_Y, R1_Z, 
                                                D2_X, D2_Y, D2_Z, R2_X, R2_Y, R2_Z,
                                                pi_max, rp_bins, boxsize, gal_weights = 1.0, 
                                                random_weights = 1.0, nthread = nthread)
    ######################################################################################################################
    return wprp_bottom20, wprp_top20


def return_box_rsd_pos(pos_x, pos_y, pos_z, vz, cosmology = Planck13, Lbox = 1000):
    pos_z_rsd = apply_zspace_distortion(true_pos = pos_z, peculiar_velocity = vz, redshift = 0, 
                                        cosmology = cosmology, Lbox = Lbox) # equal to (z + vz/100)
    gal_pos_rsd = np.array([pos_x     % Lbox,
                            pos_y     % Lbox,    
                            pos_z_rsd % Lbox]).T

    return gal_pos_rsd


def cov_to_corr(input_stat, cov):
    variances = np.diag(cov) 
    zero_element = variances == 0
    zero_elements_indices = np.where(variances == 0)[0]
    input_stat = input_stat[~zero_element]
    # bestfit = bestfit[~zero_element]
    ### Remove the ith row and column
    cov = np.delete(np.delete(cov, zero_elements_indices, axis=0), zero_elements_indices, axis=1)
    ### Normalize the covariance matrix
    stdv = np.sqrt(np.diag(cov))  # Square root of the diagonal elements
    outer_diag = np.outer(stdv, stdv)   # Outer product of the diagonal elements
    corr = cov / outer_diag
    return input_stat, corr, stdv