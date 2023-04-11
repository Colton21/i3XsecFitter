#!/bin/bash

# parameter space and variables
normmin=80
normmax=120
n_spacing=2
gammamin=210
gammamax=260
g_spacing=5

parallelcounter=0

jobtype="prop"
FLUX='astro'

# fields
dagfile="/scratch/chill/nuSQuIDS/propagate.dag"
jobname='nuSQuIDS_propagate'
jobcounter=0


# create empty dagman files
: > $dagfile

# write tablemaker dag
for ((n = $normmin; n <= $normmax; n++))
do
    if [ $(( $n % $n_spacing )) -eq 0 ]
    then
        for ((g = $gammamin; g <= $gammamax; g++))
        do
            if [ $(( $g % $g_spacing )) -eq 0 ]
            then
                jobid=$jobname\_$n\_$g\_$FLUX
                norm=$n
                gamma=$g
                flux=$FLUX
                echo $jobid
                echo JOB $jobid propagate.submit >> $dagfile
                echo VARS $jobid norm=\"${norm}\" gamma=\"${gamma}\" flux=\"${FLUX}\" type=\"${jobtype}\">> $dagfile
                ((jobcounter++))
            fi
        done
    fi
done

# be chatty

echo DAGMan will process $jobcounter jobs
echo the dagman output has been written to $dagfile
