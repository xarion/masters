package masters.benchmark.android;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;

public class BenchmarkTemperatureListener implements SensorEventListener {
  private final SensorManager sensorManager;
  private final Sensor tempSensor;
  private BenchmarkResult benchmarkResult;

  public BenchmarkTemperatureListener(SensorManager sensorManager, BenchmarkResult benchmarkResult) {
    this.benchmarkResult = benchmarkResult;
    this.sensorManager = sensorManager;
    tempSensor = sensorManager.getDefaultSensor(Sensor.TYPE_AMBIENT_TEMPERATURE);
  }

  @Override
  public void onSensorChanged(SensorEvent event) {
    benchmarkResult.setTemperature(event.values[0]);
  }

  @Override
  public void onAccuracyChanged(Sensor sensor, int accuracy) {
    // pass
  }
}
