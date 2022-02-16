import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)


def data_coord_extraction(target_pdb_file):
    """calculates vectors between CA (starting point) and CB (endpoint) and builds artificial CB for GLY based on the
        side chains of amino acids (!= GLY) before if there is an or after it if Gly is the start amino acid
        No duplicated side chain entries allowed
            input:
                target_pdb_file: pdb file with data of protein of interest
            output:
                res_data: 2d list [[Atom type, Residue 3letter, ChainID, ResidueID],...]
                res_coords: 2d list of corresponding coordinates to the new_data entries
                """
    # list of all data of the entries like [[Atom type, Residue 3letter, ChainID, ResidueID],...]
    res_data = []
    # list of all coordinates of the entries like [[x1, y1, z1],...]
    res_coords = []
    # reading the pdb file
    file = open(target_pdb_file, "r")
    for line in file:
        if "ATOM  " in line[:6]:
            line = line.strip()
            res_data += [[line[12:16].replace(" ", "").strip(), line[17:20].replace(" ", "").strip(),
                          line[21].replace(" ", "").strip(), line[22:26].replace(" ", "").strip()]]
            res_coords += [[line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]]
    file.close()

    res_data = np.asarray(res_data)
    res_coords = np.asarray(res_coords, dtype=float)
    return res_data, res_coords


one_three = {'A': "ALA",
             'C': "CYS",
             'D': "ASP",
             'E': "GLU",
             'F': "PHE",
             'G': "GLY",
             'H': "HIS",
             'I': "ILE",
             'K': "LYS",
             'L': "LEU",
             'M': "MET",
             'N': "ASN",
             'P': "PRO",
             'Q': "GLN",
             'R': "ARG",
             'S': "SER",
             'T': "THR",
             'V': "VAL",
             'W': "TRP",
             'Y': "TYR"}

three_one = {v: k for k, v in one_three.items()}


def create_sequence_from_alignment_3(file_path, num_seq, head_lines=3, lines_to_skip=2):
    """num_seq: number of sequences in the alignment
        head_lines: number of lines until the first sequence line starts
        lines_to_skip: number of lines between the blocks of sequence lines"""
    alignment = open(file_path)
    seqs = []
    for i in range(num_seq):
        seqs += [[]]
    alignment_list = list(alignment)
    alignment.close()
    # indices of which lines to use from the alignment file to reconstruct the sequences
    lines_to_use = []
    lines_encountered = 0
    lines_skipped = 0
    for i in range(head_lines, len(alignment_list)):
        if lines_encountered < num_seq:
            lines_to_use += [i]
            lines_encountered += 1
        else:
            lines_skipped += 1
        if lines_skipped == lines_to_skip:
            lines_encountered = 0
            lines_skipped = 0
    pre_sequences = np.asarray(alignment_list)[lines_to_use]

    line_count = 0
    for i in pre_sequences:
        seqs[line_count] += [i.strip().split("\t")[0].split(" ")[-1]]
        line_count += 1
        if line_count == num_seq:
            line_count = 0

    final_seq = []
    for i in seqs:
        final_seq += ["".join(i)]
    return final_seq


def read_dat(file_path):
    """file_path: path to .dat file"""
    rmsf_file = open(file_path)
    full_data = []
    colum_names = []
    for ci, i in enumerate(rmsf_file):
        line = i.strip().split(" ")
        j_data = []
        for j in line:
            if len(j) > 0:
                j_data += [j.strip()]
        if ci == 0:
            colum_names += j_data
        else:
            full_data += [j_data]
    rmsf_file.close()
    full_data_df = pd.DataFrame(full_data, columns=colum_names[1:])
    return full_data_df


def return_seq_pdb(file_path, chains=None):
    """file_path: path to pdb file
            chains: list of chains that should be used eg ['A', 'B']"""
    d, _ = data_coord_extraction(file_path)
    pdb_seq = np.unique(d[:, 1:], axis=0)
    pdb_seq_sorted = pdb_seq[np.lexsort((pdb_seq[:, 2].astype(int), pdb_seq[:, 1]))]
    if chains is not None:
        return pdb_seq_sorted[np.isin(pdb_seq_sorted[:, 1], np.asarray(chains))]
    else:
        return pdb_seq_sorted


def align_scores(sequences, seq_values):
    # sequences = ["A-VL", "AI-L", "AIV-"]
    # seq_values = [[3, 3, 3], [2, 2, 2], [1, 1, 1]]
    index_counts = np.zeros(len(sequences)).astype(int)
    scores = []
    for i in range(len(sequences)):
        scores += [[]]
    for i in range(len(sequences[0])):
        curr_pair = []
        for j in sequences:
            curr_pair += [j[i]]
        gap = np.asarray(curr_pair) != "-"
        for k in range(len(gap)):
            if gap[k]:
                scores[k] += [seq_values[k][index_counts[k]]]
            else:
                scores[k] += [np.nan]
        index_counts[gap] += 1
    return scores


def plot_scores(scores, names, save=False):
    for i in range(len(scores)):
        plt.plot(scores[i], label=names[i])
    plt.xlabel("residue index")
    plt.ylabel("RMSF")
    plt.legend()
    plt.rcParams['figure.dpi'] = 600
    if save:
        plt.savefig("aligned_rmsf", dpi=600)
    plt.show()


def seq_aligned_rmsf(pdb_paths, chain_selections, dat_paths, alignment_path, data_sel, names, head_lines=3):
    """pdb_paths: paths to the pdb files as list
        chain_selections: which chains were used as list eg [['A', 'B'], ['A', 'D'], None] None if all chains should be
         used
        dat_paths: paths to the maestro dat files as list
        alignment_path: path to the alignment file
        data_sel: column name which should be used for the RMSF data
        names: names of the proteins
        head_lines: how many lines in the alignment file till the first sequence starts eg when the first sequence
        is in the 4th line head_lines must be 3"""
    sequences = create_sequence_from_alignment_3(alignment_path, len(dat_paths), head_lines=head_lines)

    seq_values = []
    for i in range(len(pdb_paths)):
        pdb_seqi = return_seq_pdb(pdb_paths[i], chain_selections[i])
        # print(i, "".join(list(map(three_one.get, pdb_seqi[:, 0]))))
        data_seqi = read_dat(dat_paths[i])
        rmsf_seqi = np.asarray(data_seqi[data_sel], dtype=float)
        seq_values += [rmsf_seqi.tolist()]
        if not len(pdb_seqi) == len(data_seqi):
            raise ValueError("Length of the sequence from pdb file and "
                             "number of values from dat file to not match")
        insertions = np.sum(np.asarray(list(sequences[i])) == "-")
        if not len(sequences[i]) - insertions == len(pdb_seqi):
            raise ValueError("Length of the sequence used in the alignment doesn't match the length of the"
                             "sequence in the pdb file and the number of values from the dat file")

    scores = align_scores(sequences, seq_values)
    plot_scores(scores, names)


def chain_aligned_rmsf(pdb_paths, chain_selections, dat_paths, alignment_path, data_sel, names, head_lines=3):
    """pdb_paths: paths to the pdb files as list
            chain_selections: which chains were used as list eg [['A', 'B'], ['A', 'D'], None] None if all chains should be
             used
            dat_paths: paths to the maestro dat files as list
            alignment_path: path to the alignment file
            data_sel: column name which should be used for the RMSF data
            names: names of the proteins
            head_lines: how many lines in the alignment file till the first sequence starts eg when the first sequence
            is in the 4th line head_lines must be 3"""
    seq_values = []
    for i in range(len(pdb_paths)):
        pdb_seqi = return_seq_pdb(pdb_paths[i], chain_selections[i])
        data_seqi = read_dat(dat_paths[i])
        for j in np.unique(pdb_seqi[:, 1]):
            scores_per_chain = np.asarray(data_seqi[pdb_seqi[:, 1] == j][data_sel], dtype=float)
            seq_values += [scores_per_chain.tolist()]

    sequences = create_sequence_from_alignment_3(alignment_path, len(seq_values), head_lines=head_lines)

    scores = align_scores(sequences, seq_values)

    chain_names = []
    for i in range(len(names)):
        if chain_selections[i] is not None:
            for j in chain_selections[i]:
                chain_names += [names[i] + "_" + j]
        else:
            for j in np.unique(return_seq_pdb(pdb_paths[i], chain_selections[i])[:, 1]):
                chain_names += [names[i] + "_" + j]

    plot_scores(scores, chain_names)


def seq_chain_fasta(pdb_paths, chain_selections, names, seq_chain=False):
    for i in range(len(pdb_paths)):
        pdbi = return_seq_pdb(pdb_paths[i], chain_selections[i])
        if seq_chain:
            print(">" + names[i])
            print("".join(list(map(three_one.get, pdbi[:, 0]))))
        else:
            chains = np.unique(pdbi[:, 1])
            for cj, j in enumerate(chains):
                print(">" + names[i] + "_" + str(cj))
                print("".join(list(map(three_one.get, pdbi[pdbi[:, 1] == j][:, 0]))))


if __name__ == "__main__":
    names_ex = ["4ALB", "N134", "N31", "N55"]
    pdb_paths_ex = ["4alb.pdb",
                    "N134_0502.pdb",
                    "N31_0502.pdb",
                    "N55_refine_40.pdb"
                    ]
    chain_selections_ex = [['A', 'B'], ['C', 'D'], None, ['A', 'D']]  # add None if all chains should be used

    dat_paths_ex = ["C:\\Users\\~/\\PycharmProjects\\phd\\ancestor\\4albAB_report\\raw-data\\P_RMSF.dat",
                    "C:\\Users\\~/\\PycharmProjects\\phd\\ancestor\\N134_0502_report\\raw-data\\P_RMSF.dat",
                    "C:\\Users\\~/\\PycharmProjects\\phd\\ancestor\\N31_5020_report\\raw-data\\P_RMSF.dat",
                    "C:\\Users\\~/\\PycharmProjects\\phd\\ancestor\\N55_refine_40_report\\raw-data\\P_RMSF.dat"
                    ]

    alignment_path_seq_ex = "alignment_all.clustal_num"
    alignment_path_chain_ex = "alignment_per_chain.clustal_num"
    data_sel_ex = "CA"

    # get fasta of each used chain or sequence
    # seq_chain_fasta(pdb_paths_ex, chain_selections_ex, names_ex, seq_chain=True)
    # seq_aligned_rmsf(pdb_paths_ex, chain_selections_ex, dat_paths_ex, alignment_path_seq_ex, data_sel_ex, names_ex)
    chain_aligned_rmsf(pdb_paths_ex, chain_selections_ex, dat_paths_ex, alignment_path_chain_ex, data_sel_ex, names_ex)

