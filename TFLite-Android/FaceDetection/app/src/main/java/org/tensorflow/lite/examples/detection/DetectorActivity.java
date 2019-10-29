/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.app.AlertDialog;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;

import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.SparseArray;
import android.util.TypedValue;
import android.widget.Toast;

import androidx.core.util.Pair;

import com.google.android.gms.vision.Frame;
import com.google.android.gms.vision.face.Face;
import com.google.android.gms.vision.face.FaceDetector;

import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.util.Calendar;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
//  int type = super.getTypeDetect();
//  private static final int TF_OD_API_INPUT_FACE_DETECH_SIZE = 500;
//  private static final int TF_OD_API_INPUT_OBJ_DETECH_SIZE = 300;
  private static final int TF_OD_API_INPUT_FACE_NET_SIZE = 160;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;

  private static final String TF_OD_API_MODEL_FILE = "mobile_ssd_v2_float_coco.tflite";

//  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final String TF_OD_API_ANCHOR_BOX_SSD_V2 = "file:///android_asset/mobile_ssd_v2_anchor.csv";

  private static final String TF_OD_API_MODEL_FACE_NET_FILE = "face_net.tflite";
  private static final String TF_OD_API_MODEL_SVM_FILE = "svm.tflite";

  private static final String TF_OD_API_NAMES_FILE = "file:///android_asset/names.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;
  private FaceDetector face_detector;
//  private SparseArray<Face> faces = null;
//  private Classifier.faceNetOutput face_result = null;
//  private List<RectF> mappedFaces = null;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  public Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;


  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  @Override
  public void onPreviewSizeChosen(final Size size,final int rotation,final int inputSize) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = inputSize;

    face_detector = new
            FaceDetector.Builder(getApplicationContext()).setTrackingEnabled(true)
            .build();
    if(!face_detector.isOperational()){
      new AlertDialog.Builder(this).setMessage("Could not set up the face detector!").show();
      return;
    }

    if (detector != null){
      detector.close();
      detector = null;
    }
    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_INPUT_FACE_NET_SIZE,
              TF_OD_API_MODEL_FACE_NET_FILE,
              TF_OD_API_MODEL_SVM_FILE,
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_NAMES_FILE,
                  inputSize,
              TF_OD_API_IS_QUANTIZED);
      cropSize = inputSize;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(rgbFrameBitmap);
    }
    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);

            final List<Classifier.Recognition> results;
            final SparseArray<Face> faces;
//            final List<Pair<RectF,String>> mappedFaces = new LinkedList<>();
            final List<Classifier.faceNetOutput> listFaceRecognitions = new LinkedList<Classifier.faceNetOutput>();
            final List<Classifier.Recognition> mappedRecognitions = new LinkedList<Classifier.Recognition>();


            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            final long startTime = SystemClock.uptimeMillis();
            final int iType = getTypeDetect();

            detector.getOutputSsdV2(croppedBitmap);

            if (iType==TYPE_OBJECT){
                if(croppedBitmap.getWidth() != TF_OD_API_INPUT_OBJ_DETECH_SIZE)
                     return;
              results = detector.recognizeImage(croppedBitmap);
              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

              for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
                }
              }
            }
            //
            else{
              Frame frame = new Frame.Builder().setBitmap(croppedBitmap).build();
              faces = face_detector.detect(frame);

              for(int i=0; i<faces.size(); i++) {
                Face thisFace = faces.valueAt(i);
                float x1 = Math.max(thisFace.getPosition().x,0);
                float y1 = Math.max(thisFace.getPosition().y,0);
                float x2 = x1 + Math.max(thisFace.getWidth(),0);
                float y2 = y1 + Math.max(thisFace.getHeight(),0);

                RectF location = new RectF(x1,y1,x2,y2);

//                final Bitmap croppred = ImageUtils.cropFromBitmap(croppedBitmap,location,TF_OD_API_INPUT_FACE_NET_SIZE);
                if (iType==TYPE_FACE){
                  final Classifier.faceNetOutput face_result = detector.recognizeFace(croppedBitmap,location,10,TF_OD_API_INPUT_FACE_NET_SIZE);
                  listFaceRecognitions.add(face_result);

                  cropToFrameTransform.mapRect(face_result.getLocation());
                  canvas.drawRect(face_result.getLocation(), paint);
                }
                if (iType == TYPE_SAVE_CROP){
                    Pair crop = ImageUtils.cropFromBitmap2(croppedBitmap,location,10,-1);
                    Bitmap croppred = (Bitmap) crop.first;
                    RectF newRect = (RectF) crop.second;

                    if (croppred.getWidth() > TF_OD_API_MIN_SIZE_SAVE_CROP) {
                        String label = edtLabelCrop.getText().toString();
                        if (label == "")
                          label = "Unnamed";
                        Date date = Calendar.getInstance().getTime();
                        DateFormat dateFormat = new SimpleDateFormat("yyyyMMdd_hhmmss");
                        ImageUtils.saveBitmap(croppred, label
                                , String.format("Face_%s.png",dateFormat.format(date)));
                    }
                  final Classifier.faceNetOutput face_result = new Classifier.faceNetOutput("?",0F,newRect);
                  listFaceRecognitions.add(face_result);

                  cropToFrameTransform.mapRect(newRect);
                  canvas.drawRect(newRect, paint);
                }
              }

              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            }

            tracker.trackedFaces.clear();
            tracker.trackedObjects.clear();
            tracker.trackResults(mappedRecognitions,listFaceRecognitions, currTimestamp,iType);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
//                      float w = mappedFaces.get(0).right - mappedFaces.get(0).left;
                      if (listFaceRecognitions.size() > 0) {
                          final float score = listFaceRecognitions.get(0).getProba();
                          final String name = listFaceRecognitions.get(0).getName();
                          showFaceInfo("0,"+name+",Score : "+score);
//                          showFaceInfo("Face 0 : " +faceOutput.getName()+","+faceOutput.getProba());
                      }
                      showFrameInfo(previewWidth + "x" + previewHeight);
                      showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                      showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected Classifier getDetector(){
    return this.detector;
  }
  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
