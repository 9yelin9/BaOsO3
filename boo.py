#!/home/9yelin9/.local/bin/python3

import os
num_thread = 16
os.environ['OMP_NUM_THREADS'] = str(num_thread)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_thread)

import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-ml', '--mlmag', nargs='+', default=None, help='e <bins>                             : GenEnergy\n'\
																   +'p <dtype> <strain> <tol>             : GenParams\n'\
																   +'d <dtype> <strain> <tol> <bins> <ep> : GenDOS\n'\
																   +'t <dn> <mcn> <ftn> <rspn>            : Train\n'\
																   +'v <dn1> <dn2> <mcn> <ftn> <rspn>     : Validate\n')
args = parser.parse_args()                                                                     

# mlmag
if args.mlmag:
	from mlmag import mlmag
	m = mlmag.MLMag(num_thread=num_thread)

	if   args.mlmag[0] == 'e': m.GenEnergy(*args.mlmag[1:])
	elif args.mlmag[0] == 'p': m.GenParams(*args.mlmag[1:])
	elif args.mlmag[0] == 'd': m.GenDOS(*args.mlmag[1:])
	elif args.mlmag[0] == 't': m.Train(*args.mlmag[1:])
	elif args.mlmag[0] == 'v': m.Validate(*args.mlmag[1:])
	sys.exit()

parser.print_help()
