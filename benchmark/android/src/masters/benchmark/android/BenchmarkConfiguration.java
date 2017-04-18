package masters.benchmark.android;

import android.content.res.AssetManager;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class BenchmarkConfiguration {
  private String inputName;
  private String outputName;
  private int[] inputShape;
  private int[] outputShape;
  private String modelFileName;
  private int numberOfRuns;
  private long durationMilliseconds;
  private String modelName;

  public BenchmarkConfiguration(AssetManager assetManager) throws IOException {
    InputStream propertiesFile = assetManager.open("benchmark.properties");
    Properties properties = new Properties();
    properties.load(propertiesFile);
    this.initProperties(properties);
  }

  private void initProperties(Properties properties) {
    this.inputName = properties.getProperty("input_name");
    this.outputName = properties.getProperty("output_name");
    this.modelFileName = properties.getProperty("model_graph_file");
    this.numberOfRuns = Integer.valueOf(properties.getProperty("number_of_runs"));
    this.durationMilliseconds = Integer.valueOf(properties.getProperty("duration_milliseconds"));
    String inputShapeString = properties.getProperty("input_shape");
    this.inputShape = parseShape(inputShapeString);
    String outputShapeString = properties.getProperty("output_shape");
    this.outputShape = parseShape(outputShapeString);
    this.modelName = properties.getProperty("model_name");

  }

  private int[] parseShape(String shapeString) {
    String[] dimensionStrings = shapeString.split(",");
    int[] shape = new int[dimensionStrings.length];
    for (int index = 0; index < dimensionStrings.length; index++) {
      shape[index] = Integer.valueOf(dimensionStrings[index]);
    }
    return shape;
  }

  private int shapeToSize(int[] shape) {
    int size = 1;
    for (int dimension : shape) {
      size *= dimension;
    }
    return size;
  }

  public String getInputName() {
    return this.inputName;
  }

  public String getOutputName() {
    return this.outputName;
  }

  public int[] getInputShape() {
    return this.inputShape;
  }

  public String getModelFileName() {
    return this.modelFileName;
  }

  public int getNumberOfRuns() {
    return this.numberOfRuns;
  }

  public int[] getOutputShape() {
    return this.outputShape;
  }

  public int getOutputSize() {
    return shapeToSize(this.outputShape);
  }

  public int getInputSize() {
    return shapeToSize(this.inputShape);
  }

  public long getDurationMilliseconds() {
    return durationMilliseconds;
  }

  public String getModelName() {
    return modelName;
  }

  public void setModelName(String modelName) {
    this.modelName = modelName;
  }
}