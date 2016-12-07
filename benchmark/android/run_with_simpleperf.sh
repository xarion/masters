adb shell setprop security.perf_harden 0
PID=$(echo $(adb shell ps | grep masters.benchmark.android) | awk {'print $2'})
adb shell run-as masters.benchmark.android kill -15 $PID
adb shell am start -n masters.benchmark.android/masters.benchmark.android.BenchmarkActivity
PID=$(echo $(adb shell ps | grep masters.benchmark.android) | awk {'print $2'})
adb shell run-as masters.benchmark.android ./simpleperf stat -p $PID -e L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-loads,L1-icache-load-misses,L1-icache-stores,L1-icache-store-misses,dTLB-loads,dTLB-stores,iTLB-loads,iTLB-stores,branch-loads,branch-load-misses,branch-stores,branch-store-misses,node-loads,node-load-misses,node-stores,node-store-misses,node-prefetches,node-prefetch-misses,cpu-cycles,instructions,branch-instructions,branch-misses,bus-cycles,stalled-cycles-frontend,stalled-cycles-backend,cpu-clock,task-clock,page-faults,context-switches,cpu-migrations,minor-faults,major-faults,alignment-faults,emulation-faults --duration 20
adb shell setprop security.perf_harden 1

