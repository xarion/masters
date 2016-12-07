package masters.benchmark.android;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.BatteryManager;

public class BatteryReceiver extends BroadcastReceiver {

  BenchmarkRecord benchmarkRecord;

  public BatteryReceiver(BenchmarkRecord benchmarkRecord) {
    this.benchmarkRecord = benchmarkRecord;
  }

  @Override
  public void onReceive(Context context, Intent intent) {
    int level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
    benchmarkRecord.setBatteryLevel(level);

    int scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
    benchmarkRecord.setBatteryScale(scale);
  }
}

