#!/usr/bin/env bash

# create settings file
# push asset with name model.pb
# push new assets from arguments
MODELS=/Users/erdicalli/dev/workspace/masters/benchmark/models
ASSETS=/Users/erdicalli/dev/workspace/masters/benchmark/android/assets
TF_HOME=/Users/erdicalli/dev/workspace/tensorflow


echo input_name=Placeholder:0 > benchmark.properties
echo output_shape=1000 >> benchmark.properties
echo number_of_runs=-1 >> benchmark.properties
echo duration_milliseconds=20000 >> benchmark.properties
echo model_graph_file=file:///android_asset/model.pb >> benchmark.properties
echo output_name=$2 >> benchmark.properties
echo input_shape=1,$3,$3,3 >> benchmark.properties
echo model_name=$1 >> benchmark.properties

cp $MODELS/$1 $ASSETS/model.pb
cp benchmark.properties $ASSETS/benchmark.properties

HERE=$(pwd)

cd $TF_HOME
bazel mobile-install -c opt //tensorflow/masters/benchmark/android:tensorflow_demo

cd $HERE
echo 'Installed app, running benchmark'
adb push ./android/simpleperf/android/arm/simpleperf /data/local/tmp
adb shell run-as masters.benchmark.android cp /data/local/tmp/simpleperf .
adb shell setprop security.perf_harden 0
PID=$(echo $(adb shell ps | grep masters.benchmark.android) | awk {'print $2'})
adb shell run-as masters.benchmark.android kill -15 $PID
adb shell am start -n masters.benchmark.android/masters.benchmark.android.BenchmarkActivity
PID=$(echo $(adb shell ps | grep masters.benchmark.android) | awk {'print $2'})
adb shell run-as masters.benchmark.android ./simpleperf stat -p $PID -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,L1-icache-stores,L1-icache-store-misses,dTLB-loads,dTLB-stores,iTLB-loads,iTLB-stores,branch-loads,branch-load-misses,branch-stores,branch-store-misses,node-loads,node-load-misses,node-stores,node-store-misses,node-prefetches,node-prefetch-misses,cpu-cycles,instructions,branch-instructions,branch-misses,bus-cycles,stalled-cycles-frontend,stalled-cycles-backend,cpu-clock,task-clock,page-faults,context-switches,cpu-migrations,minor-faults,major-faults,alignment-faults,emulation-faults --duration 20 > $1.out
sleep 3
adb shell run-as masters.benchmark.android cat $1.log >> $1.out
adb shell setprop security.perf_harden 1

