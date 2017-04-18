package masters.benchmark.android;


import java.util.Locale;

public class BenchmarkRecord {

  private String modelName;
  private long startMilliseconds;
  private long endMilliseconds;

  private int numberOfInferences;
  private long runForMilliseconds;

  private int initialBatteryLevel = -1;
  private int finalBatteryLevel;
  private int batteryScale;

  private float initialAmbientTemperature = -1;
  private float finalAmbientTemperature;

  public void initialize(long runForMilliseconds) {
    this.runForMilliseconds = runForMilliseconds;
    numberOfInferences = 0;
    this.setStartMilliseconds(System.currentTimeMillis());
  }

  public void finalizeBenchmark() {  // function name finalize is overriding Object.finalize
    this.setEndMilliseconds(System.currentTimeMillis());
  }

  public boolean notFinished() {
    return System.currentTimeMillis() - this.startMilliseconds < this.runForMilliseconds;
  }

  public void incrementNumberOfInferences() {
    this.numberOfInferences += 1;
  }

  public void setStartMilliseconds(long startMilliseconds) {
    this.startMilliseconds = startMilliseconds;
  }

  public void setEndMilliseconds(long endMilliseconds) {
    this.endMilliseconds = endMilliseconds;
  }

  public void setBatteryLevel(int batteryLevel) {
    if (this.initialBatteryLevel == -1) {
      this.initialBatteryLevel = batteryLevel;
    }
    this.finalBatteryLevel = batteryLevel;
  }

  public void setBatteryScale(int batteryScale) {
    this.batteryScale = batteryScale;
  }

  public void setTemperature(float temperature) {
    if (this.initialAmbientTemperature == -1) {
      this.initialAmbientTemperature = temperature;
    }
    this.finalAmbientTemperature = temperature;
  }

  @Override
  public String toString() {
    return String.format(Locale.US,
        "BenchmarkResults: %s\n" +
            "total time=%d ms\n" +
            "inferences=%d\n" +
            "batteryScale=%d\n" +
            "initialBatteryLevel=%d\n" +
            "finalBatteryLevel=%d\n" +
            "initialTemperature=%.2f\n" +
            "finalTemperature=%.2f\n",
        this.modelName,
        this.endMilliseconds - this.startMilliseconds,
        this.numberOfInferences,
        this.batteryScale,
        this.initialBatteryLevel,
        this.finalBatteryLevel,
        this.initialAmbientTemperature,
        this.finalAmbientTemperature);
  }

  public String getModelName() {
    return modelName;
  }

  public void setModelName(String modelName) {
    this.modelName = modelName;
  }
}
