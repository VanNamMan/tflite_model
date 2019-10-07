/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.examples.detection.env.Logger;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {
  private static final Logger LOGGER = new Logger();

  private static final float TF_SCORE_FACE_RECOGNITION = 0.5F;
  private static final int OUTPUT_DIM_FACENET = 128;
  private static final int OUTPUT_DIM_SVM = 10;
  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_MEAN = 128.0f;
  private static final float IMAGE_STD = 128.0f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;
  private boolean isModelQuantized;
  // Config values.
  private int inputSize,inputFaceSize;
  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private Vector<String> label_names = new Vector<String>();
  private int[] intValues,initFaceValues;
  // outputLocations: array of shape [Batchsize, NUM_DETECTIONS,4]
  // contains the location of detected boxes
  private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
  private float[][] outputClasses;
  // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the scores of detected boxes
  private float[][] outputScores;
  // numDetections: array of shape [Batchsize]
  // contains the number of detected boxes

  private float[][] outputFacenetModel;
  private float[][] outputSvmModel;

  private float[] numDetections;

  private ByteBuffer imgData;
  private ByteBuffer imgFaceData;

  private Interpreter tfLite,facenet_tfLite,svm_tfLite;

  private TFLiteObjectDetectionAPIModel() {}

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */
  public static Classifier create(
      final AssetManager assetManager,
      final int inputFaceSize,
      final String facenetModelFilename,
      final String svmModelFilename,
      final String modelFilename,
      final String labelFilename,
      final String namesFilename,
      final int inputSize,
      final boolean isQuantized)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      LOGGER.w(line);
      d.labels.add(line);
    }
    br.close();

    // load label names
    actualFilename = namesFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader brNames = null;
    brNames = new BufferedReader(new InputStreamReader(labelsInput));
    while ((line = brNames.readLine()) != null) {
      LOGGER.w(line);
      d.label_names.add(line);
    }
    br.close();
    //


    d.inputSize = inputSize;
    d.inputFaceSize = inputFaceSize;

    try {
      d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
      Tensor tt = d.tfLite.getInputTensor(0);
      d.facenet_tfLite = new Interpreter(loadModelFile(assetManager, facenetModelFilename));
      d.svm_tfLite = new Interpreter(loadModelFile(assetManager, svmModelFilename));
      Tensor t = d.svm_tfLite.getOutputTensor(0);
      int i = 0;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    d.isModelQuantized = isQuantized;
    // Pre-allocate buffers.
    int numBytesPerChannel;
    if (isQuantized) {
      numBytesPerChannel = 1; // Quantized
    } else {
      numBytesPerChannel = 4; // Floating point
    }
    d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
    d.imgData.order(ByteOrder.nativeOrder());
    d.intValues = new int[d.inputSize * d.inputSize];

    d.imgFaceData = ByteBuffer.allocateDirect(1 * d.inputFaceSize * d.inputFaceSize * 3 * 4);
    d.imgFaceData.order(ByteOrder.nativeOrder());
    d.initFaceValues = new int[d.inputFaceSize * d.inputFaceSize];

    d.tfLite.setNumThreads(NUM_THREADS);
    d.outputLocations = new float[1][NUM_DETECTIONS][4];
    d.outputClasses = new float[1][NUM_DETECTIONS];
    d.outputScores = new float[1][NUM_DETECTIONS];
    d.numDetections = new float[1];

    d.facenet_tfLite.setNumThreads(NUM_THREADS);
    d.outputFacenetModel = new float[1][OUTPUT_DIM_FACENET];

    d.svm_tfLite.setNumThreads(NUM_THREADS);
    d.outputSvmModel = new float[1][OUTPUT_DIM_SVM];

    return d;
  }
  @Override
  public faceNetOutput recognizeFace(final Bitmap bitmap,final RectF location) {
    Trace.beginSection("recognizeFace");
    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(initFaceValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgFaceData.rewind();

    float mean = IMAGE_MEAN,std=IMAGE_STD;

    for (int i = 0; i < inputFaceSize; ++i) {
      for (int j = 0; j < inputFaceSize; ++j) {
        int pixelValue = initFaceValues[i * inputFaceSize + j];
        imgFaceData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);
        imgFaceData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);
        imgFaceData.putFloat(((pixelValue & 0xFF) - mean) / std);
      }
    }
    Trace.endSection(); // preprocessBitmap

    Trace.beginSection("face");
    outputFacenetModel = new float[1][OUTPUT_DIM_FACENET];
    outputSvmModel = new float[1][OUTPUT_DIM_SVM];

    Object[] inputArray = {imgFaceData};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputFacenetModel);
    Trace.endSection();

    // Run the facenet inference call.
    Trace.beginSection("run Facenet");
    facenet_tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();

    float sigma = 0, epsilon = 0.000001F;
    for(float a : outputFacenetModel[0])
      sigma+=a*a;

    sigma = Math.max(sigma,epsilon);
    sigma = (float)Math.sqrt(sigma);

    for(int i =0;i < outputFacenetModel[0].length;i++)
      outputFacenetModel[0][i] /= sigma;

    Map<Integer, Object> outputMapSVM = new HashMap<>();
    outputMapSVM.put(0, outputSvmModel);
    // Run the svm inference call.
    Trace.beginSection("run SVM");
    svm_tfLite.runForMultipleInputsOutputs(outputFacenetModel, outputMapSVM);
    Trace.endSection();

    int argmax = 0;
    float proba = 0 ;

    for (int i=0;i<outputSvmModel[0].length;i++)
      if(outputSvmModel[0][i] > proba){
        proba = outputSvmModel[0][i];
        argmax = i;
      }

    String name = "?";
      if (proba > TF_SCORE_FACE_RECOGNITION)
        name = label_names.elementAt(argmax);
    final faceNetOutput recognition = new faceNetOutput(name,proba,location);

    Trace.endSection();
    return recognition;
  }
  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

    imgData.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[i * inputSize + j];
        if (isModelQuantized) {
          // Quantized model
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        } else { // Float model
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
          imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }
      }
    }
    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");
    outputLocations = new float[1][NUM_DETECTIONS][4];
    outputClasses = new float[1][NUM_DETECTIONS];
    outputScores = new float[1][NUM_DETECTIONS];
    numDetections = new float[1];

    Object[] inputArray = {imgData};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputLocations);
    outputMap.put(1, outputClasses);
    outputMap.put(2, outputScores);
    outputMap.put(3, numDetections);
    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();

    // Show the best detections.
    // after scaling them back to the input size.
    final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
    for (int i = 0; i < NUM_DETECTIONS; ++i) {
      final RectF detection =
          new RectF(
              outputLocations[0][i][1] * inputSize,
              outputLocations[0][i][0] * inputSize,
              outputLocations[0][i][3] * inputSize,
              outputLocations[0][i][2] * inputSize);
      // SSD Mobilenet V1 Model assumes class 0 is background class
      // in label file and class labels start from 1 to number_of_classes+1,
      // while outputClasses correspond to class index from 0 to number_of_classes
      int labelOffset = 1;
      recognitions.add(
          new Recognition(
              "" + i,
              labels.get((int) outputClasses[0][i] + labelOffset),
              outputScores[0][i],
              detection));
    }

    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {}

  public void setNumThreads(int num_threads) {
    if (tfLite != null) tfLite.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLite != null) tfLite.setUseNNAPI(isChecked);
  }
}
