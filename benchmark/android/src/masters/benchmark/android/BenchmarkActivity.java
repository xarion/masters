package masters.benchmark.android;

import android.app.Activity;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.widget.TextView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class BenchmarkActivity extends Activity {

  BatteryReceiver batteryReceiver;
  BenchmarkRecord benchmarkRecord;

  private void registerBatteryReceiver() {
    this.batteryReceiver = new BatteryReceiver(benchmarkRecord);
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
    benchmarkRecord = new BenchmarkRecord();
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
          benchmark.benchmarkModel(benchmarkRecord);
          runOnUiThread(new Runnable() {
            @Override
            public void run() {
              resultTextView.setText(benchmarkRecord.toString());
            }
          });
          writeLog(benchmarkRecord);
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

  private void writeLog(BenchmarkRecord benchmarkRecord) {

    File file = new File("/data/data/masters.benchmark.android/" + benchmarkRecord.getModelName() + ".log");

    try {
//      file.createNewFile();
      FileOutputStream f = new FileOutputStream(file);
      f.write(benchmarkRecord.toString().getBytes());
      f.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

}
