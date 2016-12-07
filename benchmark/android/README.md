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

get [simpleperf](https://android.googlesource.com/platform/system/extras/+/master/simpleperf/) and put it in your phone following the README.md on the page.

## Download simpleperf to the app’s directory
Use *uname* to find the architecture on device

    $adb shell uname -m
    aarch64

"aarch64" means we should download arm64 version of simpleperf to device.

    $adb push device/arm64/simpleperf /data/local/tmp
    $adb shell run-as masters.benchmark.android cp /data/local/tmp/simpleperf .
    $adb shell run-as masters.benchmark.android chmod a+x simpleperf
    $adb shell run-as masters.benchmark.android ls -l
    -rwxrwxrwx 1 u0_a90 u0_a90 3059208 2016-01-01 10:40 simpleperf

Note that some apps use arm native libraries even on arm64 devices (We can
verify this by checking /proc/<process\_id\_of\_app>/maps). In that case, we
should use arm/simpleperf instead of arm64/simpleperf.


It's as simple as that..