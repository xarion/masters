package masters.benchmark.android;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.BatteryManager;

public class BatteryReceiver extends BroadcastReceiver {

  BenchmarkResult benchmarkResult;

  public BatteryReceiver(BenchmarkResult benchmarkResult) {
    this.benchmarkResult = benchmarkResult;
  }

  @Override
  public void onReceive(Context context, Intent intent) {
    int level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
    benchmarkResult.setBatteryLevel(level);

    int scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
    benchmarkResult.setBatteryScale(scale);
  }
}

