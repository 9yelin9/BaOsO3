#!/bin/bash
. /home/9yelin9/.bashrc

#$ -q openmp.q@phase06
#$ -pe mpi 16
#$ -j y
#$ -cwd
#$ -o log/$JOB_NAME.log

t0=$(date +%s.%N)
t0_string=$(date)

bins="bins128"
tol="_m0.10/"
fts="kb rf"
ep="0.20"

for s in `seq 0.95 0.05 1.05`
do
	for ft in $fts
	do
		epn="_ep$ep"
		end=".h5"

		boo.py -ml t kdos_a$s$tol$bins$epn$end rf gen_$ft none
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
