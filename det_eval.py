from __future__ import division
import os
import re
import sys
import argparse
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint
import numpy as np
import matplotlib.pyplot as plt
    
def process_args(args):
  parser = argparse.ArgumentParser(description='remove recoginzed string')
  
  parser.add_argument('--gt_dir',dest='gt_dir',
                      type=str,default='./gt_txts/',
                      help=('ground-truth dir')) 
  parser.add_argument('--test_list_file',dest='test_list_file',
                      type=str,default='./val_name.txt',
                      help=('test file list')) 
  parser.add_argument('--dt_dir',dest="dt_dir",
                      type=str, default='./1/dt_txts/',
                      help=('detect result dir'))
  parameters = parser.parse_args(args)
  return parameters

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def union(poly_1, poly_2):
  poly_1 = np.array(poly_1)
  poly_2 = np.array(poly_2)
  poly = np.concatenate((poly_1,poly_2))
  union_area = MultiPoint(poly).convex_hull.area
  #print 'union_area',union_area
  return union_area

def intersect(poly_1,poly_2):
  poly_1 = np.array(poly_1)
  poly_1 = MultiPoint(poly_1).convex_hull
  poly_2 = np.array(poly_2)
  poly_2 = MultiPoint(poly_2).convex_hull
  intersect_area = poly_2.intersection(poly_1).area
  return intersect_area


def det_eval(args):  
  print 'start testing...'
  iou_thresh = 0.5
  parameters = process_args(args)
  '''
  with open(parameters.test_list_file,'r') as f:
    val_list = f.readlines()
  val_names = [a.strip() for a in val_list]  
  '''
  val_names = os.listdir(parameters.gt_dir)
  
  dt_count = 0
  gt_count = 0
  gt_dict = {}
  all_dts = []
  flag_strick={}
  for i,val_name in enumerate(val_names):    
    with open(os.path.join(parameters.gt_dir, val_name)) as f:
      gt_lines = f.readlines()
      #print gt_lines
    gt_lines = [g.strip() for g in gt_lines]
    gts = [g.split(',')[0:8] for g in gt_lines]  
    gt_count += len(gt_lines)  
    #bboxs = [dt[:8] for dt in dts if (float(dt[8].strip())>0.6)] 
    gt_dict[val_name] = gts
    flag_strick[val_name] = [False]*len(gts)
    with open(os.path.join(parameters.dt_dir, 'task1_'+val_name)) as f:
      dt_lines = f.readlines()
    dt_lines = [d.strip() for d in dt_lines]
    #print 'dt_lines',dt_lines
    for dt_line in dt_lines:
      if dt_line:
        dts = dt_line.split(',')
        #print 'dts',dts 
        dts = [val_name]+dts
        all_dts.append(dts)
    dt_count += len(dt_lines)
  #print 'all_dts',all_dts
  tp = np.zeros(dt_count)
  fp = np.zeros(dt_count)
  all_dts = sorted(all_dts, key=lambda x:-float(x[9]))  ##sort by score,from high to low
    #good_dts = [dt[:8] for dt in dts if (float(dt[8].strip())>0.6)]
  for i in range(dt_count):    
    dt = all_dts[i] 
    image_name = dt[0]
    d_gts = gt_dict[image_name]
    dt_coor = [float(d) for d in dt[1:9]]    
    ious = []
    #print 'dt',dt_coor
    for j,d_gt in enumerate(d_gts):
      gt_coor = [float(g) for g in d_gt]     
      #print 'gt',gt_coor    
      rectangle_1 = []
      rectangle_2 = []
      for ii in range(0,8,2):
        rectangle_1.append([gt_coor[ii],gt_coor[ii+1]])
        rectangle_2.append([dt_coor[ii],dt_coor[ii+1]])
      union_area = union(rectangle_1, rectangle_2) 
      intersect_area = intersect(rectangle_1, rectangle_2)
      iou = intersect_area/union_area
      ious.append(iou)
    max_iou = max(ious)
    max_index = ious.index(max_iou)
    if max_iou > iou_thresh:
      if not flag_strick[image_name][max_index]:
        tp[i] = 1.
        flag_strick[image_name][max_index] = True
      else:
        fp[i] = 1.
    else:
      fp[i] = 1.
  fp = np.cumsum(fp)
  print 'fp',fp
  tp = np.cumsum(tp)
  print 'tp',tp
  rec = tp / float(gt_count)
  print rec
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  #prec = tp/(tp+fp)
  print prec
  ap = voc_ap(rec, prec)
  plt.title('Precion-Recall curve')
  plt.ylabel('Precision')
  plt.xlabel('Recall')
  plt.grid(True)
  axes = plt.gca()
  axes.set_xlim([0,1.0])
  axes.set_ylim([0,1.0])
  #axis([0,1.0,0,1.0])
  plt.plot(rec,prec,'.r-')

  plt.savefig('./1/det_PR.png')
  print 'ap',ap
  return rec, prec, ap
  

if __name__ == "__main__":
  det_eval(sys.argv[1:])
