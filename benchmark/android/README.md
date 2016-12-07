# Install

First, get the stably compiled tensorflow dependencies.

```
cd $YOUR_WORKSPACE
cd tensorflow
# connect your phone and make sure you see it on adb devices
# we're doing this to get a compiled tensorflow library
bazel mobile-install tensorflow/masters/benchmark/android:tensorflow_demo
# takes a while..
mkdir tensorflow/masters/benchmark/android/libs/
cp -R bazel-bin/tensorflow/masters/benchmark/android/_dx/tensorflow_demo/native_symlinks/* tensorflow/masters/benchmark/android/libs/

```

## Put simpleperf to the app’s directory
initialize the git submodule 

```
git submodule init
```

Use *uname* to find the architecture on device

    $adb shell uname -m
    aarch64

"aarch64" means we should use arm64 version of simpleperf.

    $adb push simpleperf/android/arm64/simpleperf /data/local/tmp
    $adb shell run-as masters.benchmark.android cp /data/local/tmp/simpleperf .
    $adb shell run-as masters.benchmark.android chmod a+x simpleperf
    $adb shell run-as masters.benchmark.android ls -l
    -rwxrwxrwx 1 u0_a90 u0_a90 3059208 2016-01-01 10:40 simpleperf

Note that some apps use arm native libraries even on arm64 devices (We can
verify this by checking /proc/<process\_id\_of\_app>/maps). In that case, we
should use arm/simpleperf instead of arm64/simpleperf.

## Install the application
Use the bazel command or Android Studio to install the application on your phone. Use `run_with_simpleperf.sh` to run simpleperf with benchmark. Make sure adb is available, and the device is visible on `adb devices` before running this.

It's as simple as that..