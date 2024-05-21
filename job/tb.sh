#!/bin/bash
. /home/9yelin9/.bashrc

#$ -q openmp.q@phase05
#$ -pe mpi 72
#$ -j y
#$ -cwd
#$ -o log/$JOB_NAME.log

t0=$(date +%s.%N)
t0_string=$(date)

#strain=$1
#if [ -z "$strain" ]; then
#	echo "No strain specified"
#	exit 
#fi

types="F A C G"
nkb=1024

strain="nost"
hf.py -i l $strain
for t in $types
do
	hf.py -i kg $strain $t
	hf.py -i kb $strain $t $nkb
	tb $strain $t $nkb
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
