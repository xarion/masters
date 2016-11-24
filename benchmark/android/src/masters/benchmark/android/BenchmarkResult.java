package masters.benchmark.android;


import java.util.Locale;

public class BenchmarkResult {
  private long startMilliseconds;
  private long endMilliseconds;


  public BenchmarkResult() {
  }

  public long getStartMilliseconds() {
    return startMilliseconds;
  }

  public void setStartMilliseconds(long startMilliseconds) {
    this.startMilliseconds = startMilliseconds;
  }

  public long getEndMilliseconds() {
    return endMilliseconds;
  }

  public void setEndMilliseconds(long endMilliseconds) {
    this.endMilliseconds = endMilliseconds;
  }

  @Override
  public String toString() {
    return String.format(Locale.US,
        "BenchmarkResult\nstartMilliseconds=%d\nendMilliseconds=%d\ntotalTime=%d",
        this.startMilliseconds, this.endMilliseconds,
        this.endMilliseconds - this.startMilliseconds);
  }
}
