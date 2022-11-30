from d4_argpars import arg_dict
from d4_generation import run_all
from d4_utils import run_dict

# root directory of all files of the project
content_root_dir = "/home/gwirn/PycharmProjects/dms"
# run new training
run_all(**arg_dict(p_dir=content_root_dir))
# rerun saved training
# run_all(**run_dict(run_name="run_name_eg_pab1_21_04_2022_123419", data_path="/path/to/log_file.csv"))
