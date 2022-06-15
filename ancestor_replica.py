import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from ancestor import clustalw_alignment_parser, read_dat, return_seq_pdb, align_scores, three_one

np.set_printoptions(threshold=sys.maxsize)

p_names = np.asarray(["4alb", "N2", "N4", "N5", "N31", "N55", "N80", "N122", "N124", "N134"])
chains = [['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'B'], ['A', 'D'], ['A', 'B'], ['A', 'B'], ['A', 'B'],
          ['C', 'D']]
chains = np.asarray(chains)
# which column of the dat file should be used
col_oi = "CA"
# how many residues should be converted to np.nan from 0:cut_from_start and from -cut_from_end:
cut_from_start = 15
cut_from_end = 25  
# index of p_names which protein should be used as baseline in the second plot and for the rmsf value search
baseline_ind = 0
# reference chain from the baseline_ind when searching for rmsf values of specific residues
chain_oi = "A"
# residues for which the rmsf value should be searched - indexed like in the pdb file
res_oi = [68, 66, 98, 41, 64, 31, 11, 13]
# where a region of residues in the alignment start in the pdb 0 indexed
start_roi = 102
end_roi = 110
# number of replicas
num_replica = 10
# whether to print the alignment or not
show_alignment = True
# whether plots should be shown
show_plots = False


def dat_path(p_name, data_num):
    # base path to dat file
    dat_base = "/media/~//D/md_sim/desmond_remd_job_{}/SID/data{}/raw-data/P_RMSF.dat".format(p_name, str(data_num))
    return dat_base


# mean RMSF values per residue for all proteins from the 10 replicas per protein
mean_res_vals = []
# mean_res_vals split per chain
res_vals_per_chain = []
# aa like in the pdb files
pdb_chain_len = []
# sequences from the pdb file as str 'AVLI'
real_seq = []
# pdb file content of selected chaind
pdb_file_content = []
for cp, p in enumerate(p_names):
    # mean over all 10 replicas
    inter_vals = []
    for i in range(1, num_replica + 1):
        inter_vals += [list(read_dat(dat_path(p, i))[col_oi])]
    # mean of all replicas of protein p
    mean_inter_vals = np.mean(np.asarray(inter_vals).astype(float), axis=0)
    mean_res_vals += [mean_inter_vals]
    # get sequences from the pdb files of specified chains
    pdb_cont = return_seq_pdb(os.path.join("~//Documents/ancestors/structures", p + ".pdb"),
                              chains[cp])
    pdb_file_content += [pdb_cont]
    # length of the first chain
    prev_len = None
    for cc, c in enumerate(np.unique(pdb_cont[:, 1])):
        # change the 3-letter code from the pdb file to one-letter code
        chain_seq = "".join(list(map(three_one.get, pdb_cont[:, 0][pdb_cont[:, 1] == c])))
        real_seq += [chain_seq]
        # print(">" + p + "_" + c)
        # print(chain_seq)
        # length of one chain
        cl = len(chain_seq)
        pdb_chain_len += [cl]
        if cc == 0:
            prev_len = cl
            res_vals_per_chain += [mean_inter_vals[:cl].tolist()]
        else:
            res_vals_per_chain += [mean_inter_vals[prev_len:].tolist()]

# each proteins chain lengths as own list
pdb_chain_len = np.split(np.asarray(pdb_chain_len), len(p_names))

# sequences from alignment with '-'
aligned_seq = clustalw_alignment_parser("~//Documents/ancestors/all_ancestor_alignment_per_chain.clustal_num",
                                        20)
if show_alignment:
    for i in aligned_seq:
        print(i)

# added np.nan to the RMSF where in the alignment "-" is present
nan_filled = align_scores(aligned_seq, res_vals_per_chain)

# RMSF values for each protein as mean over the two chains
chain_mean_rmsf = []
for ci, i in enumerate(np.split(np.arange(len(aligned_seq)), len(p_names))):
    # nan filled values of both aligned chains
    chain_mean_val = np.nanmean(nan_filled[i], axis=0)
    # change ends to np.nan
    chain_mean_val[:cut_from_start] = np.nan
    chain_mean_val[-cut_from_end:] = np.nan
    chain_mean_rmsf += [chain_mean_val.tolist()]
    plt.plot(chain_mean_val, label=p_names[ci])
x_ticks = np.arange(np.max(pdb_chain_len))[::5]
plt.xticks(x_ticks, x_ticks)
plt.xlabel("Residue index")
plt.ylabel("RMSF")
plt.legend()
if show_plots:
    plt.show()

chain_mean_rmsf = np.asarray(chain_mean_rmsf)

print("Differences in the RMSF to the selected baseline {}".format(p_names[baseline_ind]))
# which arrays are not the baseline
base_pool = np.arange(len(p_names)) != baseline_ind
all_sum = []
for ci, i in enumerate(chain_mean_rmsf[base_pool]):
    # difference to RMSF of proteins that is used as baseline
    diff_to_base = i - chain_mean_rmsf[baseline_ind]
    i_name = p_names[base_pool][ci]
    n_sum = np.nansum(diff_to_base)
    all_sum += [n_sum]
    n_median = np.nanmedian(i)

    print("{:>5} RMSF sum: {:>8} -|-  higher RMSF than baseline {:>5}%" .format(i_name, np.round(n_sum, 3),
                                                                                np.round((np.sum(diff_to_base > 0.) / 
                                                                                          len(i)) * 100, 0)))
    plt.plot(diff_to_base, label=i_name)
plt.plot(np.zeros(len(chain_mean_rmsf[0])), linestyle="dashdot", color="firebrick", label="baseline")
plt.xticks(x_ticks, x_ticks)
plt.xlabel("Residue index")
plt.ylabel("RMSF deviation from baseline")
plt.legend()
if show_plots:
    plt.show()

# average RMSF per residue
cut_vals = []
for ci, i in enumerate(res_vals_per_chain):
    i_work = i[cut_from_start: - cut_from_end]
    cut_vals += [np.sum(i_work) / len(i_work)]
cut_vals = np.asarray(cut_vals)
# get the values per protein in one column
mean_protein_rmsf = np.mean(np.asarray(np.split(cut_vals, len(p_names))), axis=1)
protein_rmsf_sort = np.argsort(mean_protein_rmsf)
print("*-*"*30)
# proteins sorted after the mean rmsf values
print("Sorted according to the mean RMSF values over the whole chain:")
print(np.round(mean_protein_rmsf, 3)[protein_rmsf_sort])
print(p_names[protein_rmsf_sort])
print("*-*"*30)
print("Region {} to {} of the sequence alignment starting positions in the real sequence".format(start_roi, end_roi))
# getting the sequence/ pdb positions of residue regions in the alignment
end_roi = end_roi + 1
chain_count = 0
for i in range(len(p_names)):
    for j in range(chains.shape[1]):
        insertions = np.asarray(list(aligned_seq[chain_count])) == "-"
        print("{:>4} Chain {} starts at residue position {:>3} and ends at {:>3} (0 indexed)".format(p_names[i],
                                                                                                     chains[i][j],
              start_roi - np.sum(insertions[:start_roi]), end_roi - np.sum(insertions[:end_roi]) - 1))
        chain_count += 1

# pdb file content of the baseline chain of interest
current_cont = np.asarray(pdb_file_content[baseline_ind])
# as many 2d ndarrays as chains in the protein and filled where '-' in seq alignment
ref_pdb = []
index_count = 0
for i in aligned_seq[0]:
    # fill the inter_pdb either with the data or with ['-', '-', '-'] if '-' in seq alignment
    if i != "-":
        ref_pdb += [current_cont[current_cont[:, 1] == chain_oi][index_count]]
        index_count += 1
    else:
        ref_pdb += [np.asarray(["-", "-", "-"])]
ref_pdb = np.asarray(ref_pdb)

# RMSF values for specific residues of all proteins
print("*-*"*30, "\nSelected residues:")
res_means = []
for r in res_oi:
    # where in the reference in the alignment the residue r of interest it located
    mask = ref_pdb[:, 2] == str(r)
    print(" ".join(ref_pdb[mask][0].astype(str).tolist()))
    inter_rvals = []
    # getting for each sequence (chain) in the alignment the value
    for i in nan_filled:
        inter_rvals += [i[mask][0]]
    res_means += [np.mean(np.split(np.asarray(inter_rvals), len(p_names)), axis=1).tolist()]
res_means = np.asarray(res_means)

for i in range(len(p_names)):
    plt.scatter(res_oi, res_means[:, i], label=p_names[i], marker="^")
plt.legend()
if show_plots:
    plt.show()
res_rmsf_sum = np.sum(res_means, axis=0)
res_rmsf_sort = np.argsort(res_rmsf_sum)
print("Sorted according to the sum of the RMSF values of the selected residues:")
print(np.round(res_rmsf_sum, 3)[res_rmsf_sort])
print(p_names[res_rmsf_sort])
print("*-*"*30)


"""
def map_bfactor(b_factor, poi, chain_oi, rmsf_norm=True):
    
    pdb_fp = "~//Documents/ancestors/rmfs_b/{}.pdb".format(poi)
    a = open(pdb_fp, "r")
    data = a.readlines()
    a.close()

    print("minimum: {:0.4f}\nmaximum:{:0.4f}".format(np.min(b_factor), np.max(b_factor)))

    name_ins = "rmsf"

    c = -1
    prev_res = None
    n = open("~//Documents/ancestors/rmfs_b/{}_{}_bfactor.pdb".format(poi, name_ins), "w+")
    for line in data:
        if "ATOM  " in line[:6]:
            name = line[17:20].replace(" ", "").strip()
            chain = line[21].replace(" ", "").strip()
            num = line[22:26].replace(" ", "").strip()
            if chain in chain_oi:
                cur_res = "".join([name, chain, num])
                if rmsf_norm:
                    if prev_res != cur_res:
                        c += 1
                        prev_res = cur_res
                else:
                    c += 1
                b_write = "{:0.2f}".format(float(b_factor[c]))
                line_list = list(line)
                for i in range(1, 7):
                    try:
                        line_list[65 - (i - 1)] = b_write[-i]
                    except IndexError:
                        line_list[65 - (i - 1)] = " "
                n.write("".join(line_list))
        else:
            n.write(line)
    n.close()

for i in range(10):
    map_bfactor(mean_res_vals[i], p_names[i], chains[i])"""