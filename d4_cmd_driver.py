from d4_argpars import arg_dict
from d4_generation import run_all
# root directory of all files of the project
content_root_dir = "~//PycharmProjects/dms"
print("*** DON'T FORGET TO SET RANDOM SEED ***")
run_all(**arg_dict(p_dir=content_root_dir))
