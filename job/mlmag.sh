#!/bin/bash
. /home/9yelin9/.bashrc

#$ -q openmp.q@phase06
#$ -pe mpi 16
#$ -j y
#$ -cwd
#$ -o log/$JOB_NAME.log

t0=$(date +%s.%N)
t0_string=$(date)

bins=128

boo.py -ml e $bins

for ep in `seq 0.1 0.1 0.3`
do
	for s in `seq 0.95 0.01 1.05`
	do
		strain="a$s"
		boo.py -ml p hdos $strain m
		boo.py -ml p kdos $strain m

		boo.py -ml d hdos $strain m $bins $ep
		boo.py -ml d kdos $strain m $bins $ep
	done
done

t1=$(date +%s.%N)
t1_string=$(date)

t=$(echo "$t1 - $t0"|bc)
h=$(echo "($t/3600)"|bc)
m=$(echo "($t%3600)/60"|bc)
s=$(echo "($t%3600)%60"|bc)

echo ""
echo "# Job ID       : $JOB_ID"
echo "# Job Name     : $JOB_NAME"
echo "# Time Start   : $t0_string"
echo "# Time End     : $t1_string"
echo "# Time Elapsed : ${h}h ${m}m ${s}s"
echo ""
