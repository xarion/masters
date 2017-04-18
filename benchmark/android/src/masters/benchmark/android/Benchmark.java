package masters.benchmark.android;

import android.content.res.AssetManager;
import android.os.Trace;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.Random;


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
  private long runForMilliseconds;
  // Pre-allocated buffers.
  private String modelName;
  private float[] floatValues;
  private float[] outputs;


  private TensorFlowInferenceInterface inferenceInterface;

  public Benchmark(AssetManager assetManager) throws IOException {
    this.initializeBenchmark(assetManager);
  }

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
    Random random = new Random();
    for (int index = 0; index < inputSize; index++) {
      floatValues[index] = random.nextFloat();
    }
    this.outputs = new float[config.getOutputSize()];

    this.numberOfRuns = config.getNumberOfRuns();
    this.runForMilliseconds = config.getDurationMilliseconds();

    String modelFileName = config.getModelFileName();
    this.modelName = config.getModelName();
    this.inferenceInterface = new TensorFlowInferenceInterface();
    return inferenceInterface.initializeTensorFlow(assetManager, modelFileName);
  }

  /**
   * Runs a benchmark for given configuration file.
   */
  public void benchmarkModel(BenchmarkRecord benchmarkRecord) {
    benchmarkRecord.setModelName(this.modelName);

    Trace.beginSection("benchmark");
    benchmarkRecord.initialize(runForMilliseconds);
    System.out.println("RUN FOR MILLISECONDS: " + runForMilliseconds);
    while (benchmarkRecord.notFinished()) {
      Trace.beginSection("singleRun");

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
      Trace.endSection(); // readNodeFloat
      Trace.endSection(); // single_run
      benchmarkRecord.incrementNumberOfInferences();
    }
    benchmarkRecord.finalizeBenchmark();
    Trace.endSection();
  }

  public void close() {
    inferenceInterface.close();
  }
}
