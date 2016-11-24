package masters.benchmark.android;


public class BenchmarkScore {
  private long startMilliseconds;
  private long endMilliseconds;


  public BenchmarkScore() {
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
}
