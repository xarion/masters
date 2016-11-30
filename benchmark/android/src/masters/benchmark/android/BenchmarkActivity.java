package masters.benchmark.android;

import android.app.Activity;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.widget.TextView;

import java.io.IOException;

public class BenchmarkActivity extends Activity {

  BatteryReceiver batteryReceiver;
  BenchmarkResult benchmarkResult;

  private void registerBatteryReceiver() {
    this.batteryReceiver = new BatteryReceiver(benchmarkResult);
    this.registerReceiver(this.batteryReceiver, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
  }

  private void unregisterBatteryReceiver() {
    try {
      this.unregisterReceiver(this.batteryReceiver);
    } catch (Exception ignored) {
    }

  }


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    benchmarkResult = new BenchmarkResult();
    registerBatteryReceiver();
    setContentView(R.layout.activity_benchmark);
    final TextView resultTextView = (TextView) findViewById(R.id.resultTextView);

    HandlerThread handlerThread = new HandlerThread("inference");
    handlerThread.start();
    Handler handler = new Handler(handlerThread.getLooper());

    try {
      final Benchmark benchmark = new Benchmark(getAssets());
      handler.post(new Runnable() {
        @Override
        public void run() {
          benchmark.benchmarkModel(benchmarkResult);
          runOnUiThread(new Runnable() {
            @Override
            public void run() {
              resultTextView.setText(benchmarkResult.toString());
            }
          });
        }
      });
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  protected void onDestroy() {
    unregisterBatteryReceiver();
    super.onDestroy();
  }
}
