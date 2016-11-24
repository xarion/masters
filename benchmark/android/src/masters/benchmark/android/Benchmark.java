package masters.benchmark.android;

import android.content.res.AssetManager;
import android.os.Trace;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;


public class Benchmark {
  static {
    System.loadLibrary("tensorflow_demo");
  }

  private static final String TAG = "Benchmark";

  // Config values.
  private String inputName;
  private String outputName;
  private String[] outputNames;
  private int[] inputShape;
  private int numberOfRuns;
  // Pre-allocated buffers.

  private float[] floatValues;
  private float[] outputs;


  private TensorFlowInferenceInterface inferenceInterface;

  /**
   * Creates a BenchmarkConfiguration and initializes a native TensorFlow session using it.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @return The native return value, 0 indicating success.
   * @throws IOException
   */
  public int initializeBenchmark(AssetManager assetManager) throws IOException {
    BenchmarkConfiguration config = new BenchmarkConfiguration(assetManager);
    this.inputName = config.getInputName();
    this.outputName = config.getOutputName();
    this.outputNames = new String[]{this.outputName};
    this.inputShape = config.getInputShape();
    int inputSize = config.getInputSize();
    this.floatValues = new float[inputSize];

    this.outputs = new float[config.getOutputSize()];

    this.numberOfRuns = config.getNumberOfRuns();

    this.inferenceInterface = new TensorFlowInferenceInterface();
    String modelFileName = config.getModelFileName();
    return inferenceInterface.initializeTensorFlow(assetManager, modelFileName);
  }

  /**
   * Runs a benchmark for given configuration file.
   */
  public void benchmarkModel() {
    Trace.beginSection("benchmark");
    BenchmarkScore benchmarkScore = new BenchmarkScore();
    benchmarkScore.setStartMilliseconds(System.currentTimeMillis());
    for (int runId = 1; runId <= this.numberOfRuns; runId += 1) {
      Trace.beginSection("single_run");

      // since we're not really interested in the result of inference,
      // we're feeding dummy data, all 0's
      Trace.beginSection("fillNodeFloat");
      inferenceInterface.fillNodeFloat(
          inputName, inputShape, floatValues);
      Trace.endSection();

      // Run the inference call.
      Trace.beginSection("runInference");
      inferenceInterface.runInference(outputNames);
      Trace.endSection();

      // Copy the output Tensor back into the output array.
      Trace.beginSection("readNodeFloat");
      inferenceInterface.readNodeFloat(outputName, outputs);
      Trace.endSection();
      Trace.endSection();
    }
    benchmarkScore.setEndMilliseconds(System.currentTimeMillis());
    Trace.endSection();
  }

  public void close() {
    inferenceInterface.close();
  }
}
