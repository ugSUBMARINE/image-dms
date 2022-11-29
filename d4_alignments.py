import sys

import numpy as np
from matplotlib import pyplot as plt

from d4_utils import aa_dict, aa_dict_pos, protein_settings

np.set_printoptions(threshold=sys.maxsize)


def clustalw(
    file_content: str, query_name: str
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[str]], np.ndarray[tuple[int], np.dtype[int]]
]:
    """reads clustalw files and returns the sequences and the query sequence position
    :parameter
        - file_content:
          read lines from the CLUSTALW file
        - query_name:
          name of the wild type protein sequence in the alignment
    :return
        - sequences:
          sequences split into all residues and "-"
        - query_pos:
          array with the position of the query sequence in sequences
    """
    # protein names and sequences
    # "----" to enable the occ_test
    proteins_so_far = ["---"]
    sequences = ["---"]
    for ci, i in enumerate(file_content):
        if ci > 2:
            line = i.strip()
            if len(line) > 0:
                s_line = line.split(" ")
                cur_ci = 0
                cur_name = None
                cur_seq = None
                for k in s_line:
                    k_line = k.strip()
                    if len(k_line) > 0:
                        if cur_ci != ci:
                            cur_ci = ci
                            cur_name = k_line
                        else:
                            cur_seq = k_line

                # whether and where a protein is already present
                occ_test = np.where(np.asarray(proteins_so_far) == cur_name)[0]
                # add id and sequence to the lists if they are not there or
                # append the sequence if a part is already there
                # excluding None sequence lines
                if cur_name.upper().isupper():
                    if len(occ_test) == 0:
                        proteins_so_far.append(cur_name)
                        sequences.append(cur_seq)
                    else:
                        sequences[occ_test[0]] += cur_seq

    # get rid of entries that are no sequences
    sequences = sequences[1:]
    proteins_so_far = np.asarray(proteins_so_far)[1:]
    sequences = np.asarray([list(i) for i in sequences if i is not None and len(i) > 1])

    # position of the wild type sequence
    query_pos = np.where(proteins_so_far == query_name)[0]
    if len(query_pos) == 0:
        raise ValueError("Protein '{}' name not found".format(str(query_name)))
    elif len(query_pos) > 1:
        raise ValueError(
            "Protein '{}' name is not unique - can't be used"
            "as wild type sequence anchor".format(str(query_name))
        )
    return sequences, query_pos


def athreem_fasta(
    data: str, query_name: str
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[str]], np.ndarray[tuple[int], np.dtype[int]]
]:
    """reads a3m and fasta formatted alignment files and returns the sequences and
    the query sequence position
    :parameter
        - data:
          read lines from the A3M file
        - query_name:
          name of the wild type protein sequence in the alignment
    :return
        - seq_arr:
          sequences of the alignment with the maximum length of the query
        - query_pos:
          where the query sequence is located in the seq_arr"""

    # all sequences as list
    sequences = []
    # where the query is positioned it the sequences
    query_pos = None
    # whether the alignment started (a3m starts with #A3M#)
    algn_started = False
    line_counter = 0
    next_seq = False
    seq_store = []
    for i in data:
        if i.startswith(">"):
            algn_started = True
            # where the query is positioned
            name = i[1:].strip()
            if name == query_name:
                query_pos = line_counter
            line_counter += 1
            next_seq = True
        elif algn_started:
            if next_seq:
                if len(seq_store) > 0:
                    sequences.append(list(seq_store[0].upper()))
                seq_store = [i.rstrip()]
                next_seq = False
            else:
                seq_store[0] += i.rstrip()
    # if query was not found
    if query_pos is None:
        raise ValueError(
            "Query '{}' not present in the alignment and can't be "
            "used as wild type sequence anchor".format(str(query_name))
        )
    # crop aligned sequences that are longer than the query sequence
    query_len = len(sequences[query_pos])
    seq_arr = []
    for i in sequences:
        i_len = len(i)
        if i_len == query_len:
            seq_arr.append(i)
        elif i_len > query_len:
            seq_arr.append(i[:query_len])

    return np.asarray(seq_arr), np.asarray([query_pos])


def calc_conservation(
    sequences: np.ndarray[tuple[int, int], np.dtype[str]],
    query_pos: np.ndarray[tuple[int], np.dtype[int]],
) -> tuple[
    np.ndarray[tuple[int, 20], np.dtype[float]], np.ndarray[tuple[int], np.dtype[int]]
]:
    """calculates the conservation of each amino acid at each sequence position
    :parameter
        - sequences:
          sequences with all the same length and removed gaps and insertions
        - query_pos:
          where the wild type sequence is located in the alignment

    :return
        - conservation_arr:
          each row specifies which amino acids are conserved at that
          sequence position and how conserved they are
        - rows:
          indexing help with indices of each sequence position"""

    # original number of sequences in the alignment
    ori_seq_num = sequences.shape[0]
    # query and sequence
    raw_query_seq = sequences[query_pos]
    # removing the query before unique
    sequences = np.delete(sequences, query_pos, 0)
    # append the unique sequences to the query
    sequences, s_inds = np.unique(sequences, axis=0, return_counts=True)
    sequences = np.append(raw_query_seq, sequences, axis=0)
    # reset query position
    query_pos = np.asarray([0])

    # get rid of non amino acid characters from BLAST
    sequences[sequences == "X"] = " "
    sequences[sequences == "B"] = " "
    sequences[sequences == "Z"] = " "
    sequences[sequences == "*"] = " "

    # get rid of gaps and insertions based on wt_seq
    sequences = np.delete(sequences, np.where(sequences[query_pos] == "-")[1], 1)
    sequences = np.delete(sequences, np.where(sequences[query_pos] == " ")[1], 1)

    # conserved residues as one letter code at each sequence position as lists
    conservation = []
    # how often a conserved residue is present at a certain position
    conservation_score = []
    for i in range(sequences.shape[1]):
        # unique conserved residues at seq position i and their number of appearance
        un, c = np.unique(sequences[:, i], return_counts=True)
        unb = np.logical_and(un != " ", un != "-", un != ".")
        un = un[unb]
        c = c[unb]
        conservation.append(un.tolist())
        conservation_score.append(c.tolist())

    # number of amino acids
    aa_len = len(aa_dict_pos)
    # sequence length
    seq_len = sequences.shape[1]
    # look-up table that gets filled with occurrences of all amino acids at each
    # sequence position
    conservation_arr = np.zeros((seq_len, aa_len))

    # populate the look-up table
    for i in range(seq_len):
        # column indices of the conserved amino acids
        i_pos = list(map(aa_dict_pos.get, conservation[i]))
        # fill look-up table
        conservation_arr[i][i_pos] = np.asarray(conservation_score[i])
    # filled look-up table with values from 0-1
    conservation_arr = conservation_arr / len(sequences)
    print(
        f"Reduced size from {ori_seq_num} to {sequences.shape[0]} unique sequences "
        "for conservation table creation"
    )
    # plt.imshow(conservation_arr)
    # plt.colorbar()
    # plt.show()

    # look-up table rows for easy indexing
    rows = np.arange(seq_len).astype(int)

    return conservation_arr, rows


def alignment_table(
    alignment_path: str, query_name: str
) -> tuple[
    np.ndarray[tuple[int, 20], np.dtype[float]], np.ndarray[tuple[int], np.dtype[int]]
]:
    """function to call the right reading function based on the input
    :parameter
        - alignment_path:
          file path to the multiple sequence alignment file
        - query_name:
          name of the wild type sequence in the alignment file
    :return
        - c_arr, rows: returns from the calc_conservation function
    """
    # read file content
    file_content = open(alignment_path, "r")
    data = file_content.readlines()
    file_content.close()

    if "CLUSTAL" in data[0].upper():
        print("CLUSTALW format detected")
        seq, qup = clustalw(data, query_name)
    elif "#A3M" in data[0].upper() or data[0].startswith(">"):
        print("A3M or FASTA format detected")
        seq, qup = athreem_fasta(data, query_name)
    else:
        seq, qup = None, None
        raise ValueError(
            "Invalid input format - please use ClustalW, A3M or" "fasta as format"
        )
    c_arr, rows = calc_conservation(seq, qup)

    return c_arr, rows


if __name__ == "__main__":
    alignment_table_ = alignment_table(
        "datasets/alignment_files/pab1_1000_experimental.clustal", "pab1"
    )
