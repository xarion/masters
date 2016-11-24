package masters.benchmark.android;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

import java.io.IOException;

public class BenchmarkActivity extends Activity {

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_benchmark);
    TextView resultTextView = (TextView) findViewById(R.id.resultTextView);

    Benchmark benchmark;
    BenchmarkResult benchmarkResult = null;

    try {
      benchmark = new Benchmark(getAssets());
      benchmarkResult = benchmark.benchmarkModel();
    } catch (IOException e) {
      e.printStackTrace();
    }

    if (benchmarkResult != null) {
      resultTextView.setText(benchmarkResult.toString());
    } else {
      resultTextView.setText(R.string.failed_benchmark_message);
    }
  }
}
