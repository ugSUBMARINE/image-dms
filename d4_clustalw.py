import numpy as np
import sys 
from matplotlib import pyplot as plt
from d4_utils import aa_dict, aa_dict_pos, protein_settings
np.set_printoptions(threshold=sys.maxsize)

def alignment_table(alignment_path, query_name):
    """creates a look-up table from an sequence alignment that tells which
        residues are conserved at which sequence position\n
        :parameter
            - alignment_path: str\n
              file path to the sequence alignment\n
            - query_name: str\n
              name of the wt sequence in the alignment file\n
        :return
            - conservation_arr: nx20 2D ndarray of floats\n
              each row specifies which amino acids are conserved at that
              sequence position and how conserved they are\n
            - rows: 1D ndarray of ints\n
              indexing help with indices of each sequence position\n"""

    file_p = open(alignment_path, "r")
    file_content = file_p.readlines()
    file_p.close()
    # protein names
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
                        proteins_so_far += [cur_name]
                        sequences += [cur_seq]
                    else:
                        sequences[occ_test[0]] += cur_seq
    
    # get rid of entries that are no sequences
    sequences = sequences[1:] 
    proteins_so_far = np.asarray(proteins_so_far)[1:]
    sequences = np.asarray([list(i) for i in sequences if i is not None and
        len(i) > 1])

    # get rid of non amino acid characters from BLAST
    sequences[sequences == "X"] = " "
    sequences[sequences == "B"] = " "
    sequences[sequences == "Z"] = " "
    sequences[sequences == "*"] = " "
 
    # get rid of gaps and insertions based on wt_seq
    query_pos = np.where(proteins_so_far == query_name)
    sequences = np.delete(sequences, np.where(sequences[query_pos] == "-")[1],
            1)
    sequences = np.delete(sequences, np.where(sequences[query_pos] == " ")[1],
            1)

    # conserved residues as one letter code at each sequence position as lists
    conservation = []
    # how often a conserved residue is present at a certain position
    conservation_score = []
    for i in range(sequences.shape[1]):
        # unique conserved residues at seq position i and their number of apperance
        un, c = np.unique(sequences[:, i], return_counts=True)
        unb = np.logical_and(un != " ", un != "-")
        un = un[unb]
        c = c[unb]
        conservation += [un.tolist()]
        conservation_score += [c.tolist()]
        
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

    # look-up table rows for easy indexing
    rows = np.arange(seq_len).astype(int)
    return conservation_arr, rows

if __name__ == "__main__":
    a = alignment_table("~//Downloads/avgfp_1000_experimental.clustal",
            "avgfp")

