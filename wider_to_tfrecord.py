import tensorflow as tf
import numpy
import cv2
import os
import hashlib
import io

import config
from utils import dataset_util


def print_error(e):
  print(e)
  import traceback
  print(traceback.format_exc())


class FaceNumIsNone(Exception):
  pass


class FileNameIsNone(Exception):
  pass


def parse_test_example(f, images_path):
  height = None  # Image height
  width = None  # Image width
  filename = None  # Filename of the image. Empty if image is not from file
  encoded_image_data = None  # Encoded image bytes
  image_format = b'jpeg'  # b'jpeg' or b'png'

  filename = f.readline().rstrip()
  if not filename:
    raise FileNameIsNone()

  filepath = os.path.join(images_path, filename)

  image_raw = cv2.imread(filepath)
  if config.RESIZE:
    image_raw = cv2.resize(image_raw, (config.RESIZE, config.RESIZE))

  is_success, buffer = cv2.imencode(".jpg", image_raw)
  encoded_image_data = buffer.tobytes()
  # encoded_image_data = io.BytesIO(buffer)
  # encoded_image_data = open(filepath, "rb").read()
  # key = hashlib.sha256(encoded_image_data).hexdigest()
  key = ''

  height, width, channel = image_raw.shape

  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(int(height)),
    'image/width': dataset_util.int64_feature(int(width)),
    'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
    'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    # 'image/array': dataset_util.float_list_feature(
    #   (cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB) / 255.).flatten().tolist()),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
  }))

  return tf_example


def parse_example(f, images_path):
  height = None  # Image height
  width = None  # Image width
  filename = None  # Filename of the image. Empty if image is not from file
  encoded_image_data = None  # Encoded image bytes
  image_format = b'jpeg'  # b'jpeg' or b'png'

  xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
  ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
  classes_text = []  # List of string class name of bounding box (1 per box)
  classes = []  # List of integer class id of bounding box (1 per box)
  poses = []
  truncated = []
  difficult_obj = []
  raw_all_annot = []

  filename = f.readline().rstrip()
  if not filename:
    raise FileNameIsNone()

  filepath = os.path.join(images_path, filename)
  if os.path.isfile(filepath) == False:
    raise IOError()

  face_num = int(f.readline().rstrip())
  if not face_num:
    raise FaceNumIsNone()

  for i in range(face_num):
    annot = f.readline().rstrip().split()
    if not annot:
      raise Exception()

    raw_all_annot.append(annot)

  image_raw = cv2.imread(filepath)
  if image_raw is None:
    raise IOError()
  original_height, original_width, original_channel = image_raw.shape
  # aspect_ratio = original_width / original_height
  # if aspect_ratio < .9 or aspect_ratio > 1.1:
  #   # image looses too much info if not square cropped
  #   bg_i = -1  # biggest
  #   bg_wh = 0
  #   for i in range(len(raw_all_annot)):
  #     annot = raw_all_annot[i]
  #     if float(annot[2]) > 25.0 and float(annot[3]) > 30.0:
  #       sum = float(annot[2]) + float(annot[3])
  #       if sum > bg_wh:
  #         bg_i = i
  #         bg_wh = sum
  #
  #   bg_annot = raw_all_annot[bg_i]
  #   bg_box_center = (float(bg_annot[0]) + float(bg_annot[2]) / 2, float(bg_annot[1]) + float(bg_annot[3]) / 2)
  #   min_d_start = min(bg_box_center[0], bg_box_center[1])
  #   min_d = min_d_start
  #   dx_end = original_width - bg_box_center[0]
  #   dy_end = original_height - bg_box_center[1]
  #   min_d_end = min(dx_end, dy_end)
  #
  #   if min_d_end < min_d_start:
  #     min_d = min_d_end
  #
  #   new_x_axis = bg_box_center[0] - min_d
  #   new_y_axis = bg_box_center[1] - min_d
  #   new_w = bg_box_center[0] + min_d
  #   new_h = bg_box_center[1] + min_d
  #   image_raw = image_raw[new_y_axis:new_h, new_x_axis:new_w]
  #   raw_all_annot = [
  #     annot for annot in raw_all_annot if
  #     float(bg_annot[0]) + float(bg_annot[2]) <= new_w and
  #     float(bg_annot[1]) + float(bg_annot[3]) <= new_h
  #   ]
  #   raw_all_annot = [
  #     [float(annot[0]) - new_x_axis, float(annot[1]) - new_y_axis, annot[2], annot[3]] for annot in raw_all_annot
  #   ]

  if config.RESIZE:
    image_raw = cv2.resize(image_raw, (config.RESIZE, config.RESIZE))

  is_success, buffer = cv2.imencode(".jpg", image_raw)
  encoded_image_data = buffer.tobytes()
  # encoded_image_data = io.BytesIO(buffer)
  # encoded_image_data = open(filepath, "rb").read()
  # key = hashlib.sha256(encoded_image_data).hexdigest()
  key = ''

  height, width, channel = image_raw.shape

  scaleW = width / original_width
  scaleH = height / original_height

  for i in range(len(raw_all_annot)):
    annot = raw_all_annot[i]

    # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
    if float(annot[2]) > 25.0 and float(annot[3]) > 30.0:
      xmins.append(max(0.005, (float(annot[0]) * scaleW) / width))
      ymins.append(max(0.005, (float(annot[1]) * scaleH) / height))
      xmaxs.append(min(0.995, ((float(annot[0]) + float(annot[2])) * scaleW) / width))
      ymaxs.append(min(0.995, ((float(annot[1]) + float(annot[3])) * scaleH) / height))
      classes_text.append(b'face')
      classes.append(1)
      poses.append("front".encode('utf8'))
      truncated.append(int(0))

  if len(classes) == 0:
    return None

  to_str = lambda l: (str(x) for x in l)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(int(height)),
    'image/width': dataset_util.int64_feature(int(width)),
    'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
    'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    # 'image/array': dataset_util.float_list_feature(
    #   (cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB) / 255.).flatten().tolist()),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'image/object/bbox/encoded': dataset_util.bytes_feature(
      ','.join(to_str(xmins + xmaxs + ymins + ymaxs)).encode('utf-8')
    ),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    'image/object/difficult': dataset_util.int64_list_feature(int(0)),
    'image/object/truncated': dataset_util.int64_list_feature(truncated),
    'image/object/view': dataset_util.bytes_list_feature(poses),
  }))

  return tf_example


def run(images_path, description_file, output_path, no_bbox=False):
  f = open(description_file)
  writer = tf.io.TFRecordWriter(output_path)

  i = 0

  print("Processing {}".format(images_path))
  while True:
    try:
      if no_bbox:
        tf_example = parse_test_example(f, images_path)
      else:
        tf_example = parse_example(f, images_path)

      if tf_example is None:
        continue

      writer.write(tf_example.SerializeToString())
      i += 1

    except FileNameIsNone:
      break
    except FaceNumIsNone:
      continue
    except IOError:
      continue
    except Exception:
      raise

  writer.close()

  print("Correctly created record for {} images\n".format(i))


def main(unused_argv):
  # Training
  try:
    if config.TRAIN_WIDER_PATH is not None:
      images_path = os.path.join(config.TRAIN_WIDER_PATH, "images")
      description_file = os.path.join(config.GROUND_TRUTH_PATH, "wider_face_train_bbx_gt.txt")
      output_path = os.path.join(config.OUTPUT_PATH, "train.tfrecord")
      if os.path.isfile(output_path) is False:
        run(images_path, description_file, output_path)
  except Exception as e:
    print_error(e)

  # Validation
  try:
    if config.VAL_WIDER_PATH is not None:
      images_path = os.path.join(config.VAL_WIDER_PATH, "images")
      description_file = os.path.join(config.GROUND_TRUTH_PATH, "wider_face_val_bbx_gt.txt")
      output_path = os.path.join(config.OUTPUT_PATH, "val.tfrecord")
      if os.path.isfile(output_path) is False:
        run(images_path, description_file, output_path)
  except Exception as e:
    print_error(e)

  # Testing. This set does not contain bounding boxes, so the tfrecord will contain images only
  try:
    if config.TEST_WIDER_PATH is not None:
      images_path = os.path.join(config.TEST_WIDER_PATH, "images")
      description_file = os.path.join(config.GROUND_TRUTH_PATH, "wider_face_test_filelist.txt")
      output_path = os.path.join(config.OUTPUT_PATH, "test.tfrecord")
      if os.path.isfile(output_path) is False:
        run(images_path, description_file, output_path, no_bbox=True)
  except Exception as e:
    print_error(e)


if __name__ == '__main__':
  tf.compat.v1.app.run()
