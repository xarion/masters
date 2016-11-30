package masters.benchmark.android;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.widget.TextView;

import java.io.IOException;

public class BenchmarkActivity extends Activity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
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
          BenchmarkResult benchmarkResult = null;
          benchmarkResult = benchmark.benchmarkModel();
          if (benchmarkResult != null) {
            resultTextView.setText(benchmarkResult.toString());
          } else {
            resultTextView.setText(R.string.failed_benchmark_message);
          }
        }
      });

    } catch (IOException e) {
      e.printStackTrace();
    }


  }


}
