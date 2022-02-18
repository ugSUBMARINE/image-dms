import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)


def data_coord_extraction(target_pdb_file):
    """Reads the content of a given pdb file\n
        input:
            target_pdb_file: pdb file with data of protein of interest\n
        output:
            res_data: 2d list [[Atom type, Residue 3letter, ChainID, ResidueID],...]\n
            res_coords: 2d list of corresponding coordinates to the new_data entries\n
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


def clustalw_alignment_parser(file_path, num_seq, head_lines=3, lines_to_skip=2):
    """extracts each aligned sequence of a sequence alignment and joins it to return each aligned sequence as a string\n
        input:
            file_path: path to the sequence alignment as str\n
            num_seq: number of sequences in the alignment as int\n
            head_lines: number of lines until the first sequence line starts as int\n
            lines_to_skip: number of lines between the blocks of sequence lines as int\n
        :return
            final_seq: numpy array of strings like ["A-VL", "AI-L", "AIV-"] """
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
    return np.asarray(final_seq)


def read_dat(file_path):
    """reads .dat file from SimulationInteractionsDiagram analysis of Schrödinger Desmond\n
        input:
            file_path: path to .dat file as str\n
        :return
            full_data_df: .dat file as a pandas dataframe"""
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
    """ get protein sequence as present in the pdb file in file_path\n
        input:
            file_path: path to pdb file as str\n
            chains: list of chains that should be used eg ['A', 'B'] - if None all chains will be used\n
        :return
            pdb_seq_sorted: numpy array that contains [[Residue 3letter, ChainID, ResidueID],...] of either
             all chains if chains=None or of the specified chains in chains
        """
    d, _ = data_coord_extraction(file_path)
    pdb_seq = np.unique(d[:, 1:], axis=0)
    pdb_seq_sorted = pdb_seq[np.lexsort((pdb_seq[:, 2].astype(int), pdb_seq[:, 1]))]
    if chains is not None:
        return pdb_seq_sorted[np.isin(pdb_seq_sorted[:, 1], np.asarray(chains))]
    else:
        return pdb_seq_sorted


def align_scores(sequences, seq_values):
    """inserts np.Nan in list of seq_values when '-' in sequence\n
        input:
            sequences: list of sequences from an alignment as strings eg ["A-VL", "AI-L", "AIV-"]\n
            seq_values: list of lists with corresponding scores to the sequences eg [[3, 3, 3], [2, 2, 2], [1, 1, 1]]\n
        :return
            scores: numpy array(s) with added np.Nan where a '-' is present in the sequence """
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
    return np.asarray(scores)


def plot_scores(scores, names, save=False, y_axis="Protein RMSF (\u212B)", x_axis="residue index", add_top=0.5,
                add_bottom=-0.2):
    """plots given scores\n
        input:
            scores: list or list of lists that should be displayed\n
            names: list with names according to the scores\n
            y_axis: ylabel as str\n
            x_axis: xlabel as str\n
            add_top: top limit of the ylim as float/int\n
            add_bottom: bottom limit of the ylim as float/int\n
        :return
            None"""
    if len(scores.shape) > 1 and scores.shape[0] != 1:
        for i in range(len(names)):
            plt.plot(scores[i], label=names[i])
    else:
        plt.plot(scores[0], label=names[0])
    plt.plot(np.zeros(len(scores[0])), linestyle="dashdot", color="firebrick")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.ylim(np.min(scores[~np.isnan(scores)]) + add_bottom, np.max(scores[~np.isnan(scores)]) + add_top)
    plt.rcParams['figure.dpi'] = 600
    if save:
        plt.savefig("aligned_rmsf", dpi=600)
    plt.show()


def seq_aligned_rmsf(pdb_paths, chain_selections, dat_paths, alignment_path, data_sel, names, head_lines=3, cut=10,
                     selection=None):
    """creates a plot of aligned RMSF values for each protein sequence based on a given sequence alignment\n
        input:
            pdb_paths: paths to the pdb files as list\n
            chain_selections: which chains were used as list eg [['A', 'B'], ['A', 'D'], None] None if all chains
            should be used\n
            dat_paths: paths to the maestro dat files as list\n
            alignment_path: path to the alignment file as str\n
            data_sel: column name which should be used for the RMSF data as str\n
            names: names of the proteins as list of strings\n
            head_lines: how many lines in the alignment file till the first sequence starts eg when the first sequence
            is in the 4th line head_lines must be 3\n
            selection: list with indices of the sequences of choice as given in chain_selection as list of integers or
            None if all shoul be used\n
        :return
            None"""
    sequences = clustalw_alignment_parser(alignment_path, len(dat_paths), head_lines=head_lines)

    seq_values = []
    for i in range(len(pdb_paths)):
        pdb_seqi = return_seq_pdb(pdb_paths[i], chain_selections[i])
        data_seqi = read_dat(dat_paths[i])
        rmsf_seqi = np.asarray(data_seqi[data_sel], dtype=float)
        if cut > 0:
            rmsf_seqi[:cut] = np.nan
            rmsf_seqi[-cut:] = np.nan
        seq_values += [rmsf_seqi.tolist()]
        if not len(pdb_seqi) == len(data_seqi):
            raise ValueError("Length of the sequence from pdb file and "
                             "number of values from dat file to not match")
        insertions = np.sum(np.asarray(list(sequences[i])) == "-")
        if not len(sequences[i]) - insertions == len(pdb_seqi):
            raise ValueError("Length of the sequence used in the alignment doesn't match the length of the"
                             "sequence in the pdb file and the number of values from the dat file")

    if selection is None:
        selection = np.arange(len(seq_values))
    else:
        selection = np.asarray(selection)

    scores = align_scores(sequences[selection], np.asarray(seq_values, dtype=object)[selection])
    plot_scores(scores, np.asarray(names)[selection])


def chain_aligned_rmsf(pdb_paths, chain_selections, dat_paths, alignment_path, data_sel, names, head_lines=3, cut=10,
                       selection=None, plots="", baseline_ind=None, res_sele_start=None, res_sele_end=None):
    """creates a plot of aligned RMSF values for each chain in a protein based on a given sequence alignment or
        plots the difference to a given baseline\n
        input:
            pdb_paths: paths to the pdb files as list\n
            chain_selections: which chains were used as list eg [['A', 'B'], ['A', 'D'], None] None if all chains should
            be used\n
            dat_paths: paths to the maestro dat files as list\n
            alignment_path: path to the alignment file as str\n
            data_sel: column name which should be used for the RMSF data as str\n
            names: names of the proteins as list of strings\n
            head_lines: how many lines in the alignment file till the first sequence starts eg when the first sequence
            is in the 4th line (1 indexed) head_lines must be 3\n
            selection: list of integers or None with indices of the chains of choice as given in chain_selection or all
            get plotted if None\n
            plots:
                - 'mean': plots mean RMSF of chains per residues per protein\n
                - 'all': plots RMSF of all chains separate\n
                - 'diff': plots RMSF difference of all proteins except the names[baseline_ind] protein against the
                - '""': no plots
                baseline protein\n
                    - baseline_ind: int or None if None mean of all chains gets used as baseline otherwise the with the
                    names[baseline_ind] specified protein is used as baseline\n
                    - refers to proteins specified in selection len(selection) > baseline_ind-1 is required\n
        :return
            None"""

    names = np.asarray(names)

    seq_values = []
    for i in range(len(pdb_paths)):
        pdb_seqi = return_seq_pdb(pdb_paths[i], chain_selections[i])
        data_seqi = read_dat(dat_paths[i])
        for j in np.unique(pdb_seqi[:, 1]):
            scores_per_chain = np.asarray(data_seqi[pdb_seqi[:, 1] == j][data_sel], dtype=float)
            if cut > 0:
                scores_per_chain[:cut] = np.nan
                scores_per_chain[-cut:] = np.nan
            seq_values += [scores_per_chain.tolist()]

    chain_names = []
    for i in range(len(names)):
        if chain_selections[i] is not None:
            for j in chain_selections[i]:
                chain_names += [names[i] + "_" + j]
        else:
            for j in np.unique(return_seq_pdb(pdb_paths[i], chain_selections[i])[:, 1]):
                chain_names += [names[i] + "_" + j]

    if selection is None:
        selection = np.arange(len(seq_values))
    else:
        selection = np.asarray(selection)

    sequences = clustalw_alignment_parser(alignment_path, len(seq_values), head_lines=head_lines)[selection]

    scores = align_scores(sequences, np.asarray(seq_values, dtype=object)[selection])

    chains = int(len(seq_values) / len(names))
    chain_means = []
    starts = np.arange(len(scores))[::chains]
    name_selection = []
    for i in range(len(starts)):
        name_selection += [chain_names[selection[starts[i]]][:-2]]
        if i < len(starts) - 1:
            mean_chain = np.mean(np.asarray(scores[starts[i]: starts[i + 1]]), axis=0)
        else:
            mean_chain = np.mean(np.asarray(scores[starts[i]: starts[i] + chains + 1]), axis=0)
        chain_means += [mean_chain.tolist()]
    chain_means = np.asarray(chain_means, dtype=float)

    if res_sele_start is not None and res_sele_end is not None:
        for i in range(len(sequences)):
            seqi = np.asarray(list(sequences[i]))
            insertion = seqi == "-"
            insertion_in_selection = insertion[res_sele_start:res_sele_end]
            res_sele_end = res_sele_end + np.sum(insertion_in_selection)
            insertion_before_selection = seqi[: res_sele_start] == "-"
            num_ibs = np.sum(insertion_before_selection)
            print("Sequence of interest for {} starts at {} and end at {}".format(np.asarray(chain_names)[selection][i],
                                                                                  str(res_sele_start - num_ibs),
                                                                                  str(res_sele_end - num_ibs)))

    if baseline_ind is None:
        baseline = np.mean(chain_means, axis=0)
        test_chains = chain_means
        diff_names = name_selection
    else:
        baseline = chain_means[baseline_ind]
        not_baseline = np.invert(np.arange(len(chain_means)) == baseline_ind)
        test_chains = chain_means[not_baseline]
        diff_names = np.asarray(name_selection)[not_baseline]

    diff = test_chains - baseline

    for i, j in zip(diff_names, np.sum(diff > 0, axis=1) / diff.shape[1] * 100):
        print("{}% of {} have a higher RMSF than the baseline".format(str("%.2f" % j), i))

    if plots == "mean":
        plot_scores(chain_means, name_selection)
    elif plots == "diff":
        plot_scores(diff, diff_names, y_axis="difference to baseline (\u212B)")
    elif plots == "all":
        plot_scores(scores, np.asarray(chain_names)[selection])
    elif plots == "":
        pass
    else:
        raise ValueError("given key '{}' for plots doesn't exist".format(plots))


def seq_chain_fasta(pdb_paths, chain_selections, names, seq_chain=False):
    """creates fasta formatted string of the sequences given in pdb_paths\n
        input:
            pdb_paths: list of file_path to the pdb files\n
            chain_selections: list of lists specifying the chain of interest eg [['A', 'B'], ['C', 'D']]\n
            names: list of strings that contain the names of the proteins in pdb_paths\n
            seq_chain:
                if True fasta formats of all sequences get created\n
                if False fasta formats of all chains of all sequences get created\n
        :return
            sequences: list of strings containing the sequences\n
            """
    sequences = []
    for i in range(len(pdb_paths)):
        pdbi = return_seq_pdb(pdb_paths[i], chain_selections[i])
        if seq_chain:
            print(">" + names[i])
            seq = "".join(list(map(three_one.get, pdbi[:, 0])))
            print(seq)
            sequences += [seq]
        else:
            chains = np.unique(pdbi[:, 1])
            for cj, j in enumerate(chains):
                print(">" + names[i] + "_" + str(cj))
                seq = "".join(list(map(three_one.get, pdbi[pdbi[:, 1] == j][:, 0])))
                print(seq)
                sequences += [seq]
    return sequences


def plot_rmsd(rmsd_dat_paths, names, data_sel, indices=None):
    """plots RMSD plots of data in rmsd_dat_paths\n
        input:
            rmsd_dat_paths: list of paths to the .dat files\n
            names: list of names corresponding to the .dat files in rmsd_dat_path\n
            data_sel: string with the name of the data colum of the .dat file that should be used\n
            indices: which data should be plotted - if None all get plotted\n
        :return
            None
            """
    if indices is None:
        indices = np.arange(len(names))
    names = np.asarray(names)[np.asarray(indices)]
    rmsd_values = []
    for i in range(len(rmsd_dat_paths)):
        if i in indices:
            rmsd_values += [read_dat(rmsd_dat_paths[i])["Prot_" + data_sel].values.tolist()]
    plot_scores(np.asarray(rmsd_values, dtype=float), names=names, y_axis="Protein RMSD (\u212B)", x_axis="time (nsec)")


if __name__ == "__main__":
    # IMPORTANT !!! order needs to be the same everywhere !!!
    data_sel_ex = "CA"
    names_ex = ["4ALB", "N134", "N31", "N55"]
    chain_selections_ex = [['A', 'B'], ['C', 'D'], None, ['A', 'D']]

    pdb_paths_ex = ["~//Documents/ancestors/4alb.pdb",
                    "~//Documents/ancestors/N134_0502.pdb",
                    "~//Documents/ancestors/N31_0502.pdb",
                    "~//Documents/ancestors/N55_refine_40.pdb"
                    ]
    rmsf_dat_paths_ex = ["/media/~//D/programs/schrodinger/SID_reports/4albAB_report/raw-data/P_RMSF.dat",
                         "/media/~//D/programs/schrodinger/SID_reports/N134_0502_report/raw-data/P_RMSF.dat",
                         "/media/~//D/programs/schrodinger/SID_reports/N31_5020_report/raw-data/P_RMSF.dat",
                         "/media/~//D/programs/schrodinger/SID_reports/N55_refine_40_report/raw-data/P_RMSF.dat"
                         ]
    rmsd_dat_paths_ex = ["/media/~//D/programs/schrodinger/SID_reports/4albAB_report/raw-data/PL_RMSD.dat",
                         "/media/~//D/programs/schrodinger/SID_reports/N134_0502_report/raw-data/PL_RMSD.dat",
                         "/media/~//D/programs/schrodinger/SID_reports/N31_5020_report/raw-data/PL_RMSD.dat",
                         "/media/~//D/programs/schrodinger/SID_reports/N55_refine_40_report/raw-data/PL_RMSD.dat"
                         ]

    alignment_path_seq_ex = "~//Documents/ancestors/alignment_all.clustal_num"
    alignment_path_chain_ex = "~//Documents/ancestors/alignment_per_chain.clustal_num"

    # get fasta of each used chain or sequence
    # seq_chain_fasta(pdb_paths_ex, chain_selections_ex, names_ex, seq_chain=True)

    # plot aligned RMSF scores per sequence
    # seq_aligned_rmsf(pdb_paths_ex, chain_selections_ex, rmsf_dat_paths_ex, alignment_path_seq_ex, data_sel_ex,
    #                  names_ex, selection=None)

    # plot aligned RMSF scores per chain
    chain_aligned_rmsf(pdb_paths_ex, chain_selections_ex, rmsf_dat_paths_ex, alignment_path_chain_ex, data_sel_ex,
                       names_ex, selection=None, cut=0, plots="diff", baseline_ind=0, res_sele_start=79,
                       res_sele_end=89)

    # plot RMSD of different chains
    # plot_rmsd(rmsd_dat_paths_ex, names_ex, data_sel_ex, [0, 1])
