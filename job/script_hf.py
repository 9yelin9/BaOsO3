#!/home/9yelin9/.local/bin/python3

import os
import re
import argparse
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
#parser.add_argument('-st', '--strain', nargs='+', help='<strain1> <strain2> ...')
parser.add_argument('-u', '--U', nargs='+', type=float, help='<step> <stop>')

parser.add_argument('--band', type=int,   help='<Nkb>')
parser.add_argument('--dos',  type=float, help='<ep>')
args = parser.parse_args()                                                                     

fd = open('job/default.txt', 'r')
save = 'dU%.1f_UF%.1f' % (args.U[0], args.U[1])
os.makedirs('output/%s' % save, exist_ok=True)

if   args.band: save_job = save + '_band_Nk%d' % args.band
elif args.dos:  save_job = save + '_dos_ep%.2f' % args.dos
else:           save_job = save

os.makedirs('job/%s' % save_job, exist_ok=True)
os.makedirs('log/%s' % save_job, exist_ok=True)

st = 'nost'
Q = 5

for t in ['F0', 'A0', 'A2', 'A5', 'C0', 'C1', 'C6', 'G0']:
	for j in [0, 0.1, 0.2]:
		fn = 'job/%s/%s_%s_JU%.2f.sh' % (save_job, st, t, j)
		f = open(fn, 'w')

		for line in fd:
			if re.search('#[$] -q', line):
				f.write('#$ -q mpi.q@phase0%d\n' % Q)
			elif re.search('#[$] -o', line):
				f.write('#$ -o log/%s/$JOB_NAME.log\n' % save_job)
			elif re.search('###', line):
				if re.search('F', t): f.write('for n in `seq 0.1 0.1 5.9`\ndo\n')
				else:                 f.write('for n in `seq 0.2 0.2 11.8`\ndo\n')

				if args.band or args.dos:
					f.write('\tfor u in `seq %.1f -%.1f 0`\n\tdo\n' % (args.U[1], args.U[0]))

					f.write('\t\tU="_U$u"\n')
					if   args.band: f.write('\t\thf N$n$U %s %s %s %.2f $n $u %d\n\tdone\ndone\n'   % (save, st, t, j, args.band))
					elif args.dos:  f.write('\t\thf N$n$U %s %s %s %.2f $n $u %.2f\n\tdone\ndone\n' % (save, st, t, j, args.dos))
				else:
					f.write('\thf init %s %s %s %.2f $n %.1f\n' % (save, st, t, j, args.U[1]))	
					f.write('\tfor u in `seq %.1f -%.1f 0`\n\tdo\n' % (args.U[1]-args.U[0], args.U[0]))

					f.write('\t\tU="_U$(echo "$u+%.1f"|bc)"\n' % args.U[0])
					f.write('\t\thf N$n$U %s %s %s %.2f $n $u\n\tdone\ndone\n' % (save, st, t, j))
			else: f.write(line)

		print(fn)
		f.close()
		fd.seek(0)
fd.close()
