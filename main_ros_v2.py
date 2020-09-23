# Author: Xinshuo Weng
# Modified by: Javier del Egido

from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import os, numpy as np, time, sys
from AB3DMOT_libs.model import AB3DMOT
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing

import rospy
from pyquaternion import Quaternion
from t4ac_perception_msgs.msg import Object_kitti_list, Object_kitti
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import MarkerArray, Marker

class rosTracker(): 

  def callback(self, msg):
    start_time_callback = time.time()
    #Get all detected objects
    seq_dets = np.empty((0,15), float)
    frame = msg.header.seq
    for obj in msg.object_list:
      obj_seq_dets = np.array([frame, 
                               float(obj.type.data),
                               obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3],
                               obj.score,
                               obj.dims[0], obj.dims[1], obj.dims[2],
                               obj.loc[0],  obj.loc[1],  obj.loc[2],
                               obj.rot,
                               obj.alpha]).reshape((1,15))
      seq_dets = np.append(seq_dets, obj_seq_dets, axis=0)

    #Update tracking module

    dets = seq_dets[:,7:14]
    ori_array = seq_dets[:,-1].reshape((-1, 1))
    other_array = seq_dets[:,1:7]
    additional_info = np.concatenate((ori_array, other_array), axis=1)
    dets_all = {'dets': dets, 'info': additional_info}
    self.total_frames += 1
    start_time_AB3DMOT = time.time()
    trackers = self.mot_tracker.update(dets_all)
    cycle_time = time.time() - start_time_AB3DMOT
    print("AB3DMOT time", cycle_time)
    self.total_time += cycle_time
    save_trk_dir = os.path.join("./results/pcdet_KITTI", 'trk_withid', "0016"); mkdir_if_missing(save_trk_dir)
    save_trk_file = os.path.join(save_trk_dir, '%06d.txt' % frame)
    #save_trk_file = open(save_trk_file, 'w')
    for d in trackers:              #d = x, y, z, theta, l, w, h, ID, other info, confidence
      bbox3d_tmp = d[0:7]
      id_tmp = d[7]
      ori_tmp = d[8]
      type_tmp = self.det_id2str[d[9]]
      bbox2d_tmp_trk = d[10:14]
      conf_tmp = d[14]
      '''
      str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
        bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
        bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], 
        conf_tmp, id_tmp)
      save_trk_file.write(str_to_srite)
      '''
      '''
      str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, 
        type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
        bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], 
        conf_tmp)
      eval_file.write(str_to_srite)
    eval_file.close()
    '''
    #save_trk_file.close()
    #Publish rviz tracking markers
    MarkerArray_list = self.tracking_to_rviz(trackers, msg)
    Kitti_list       = self.tracking_to_visualizer(trackers, msg)
    self.markerPublisher.publish(MarkerArray_list)
    self.kittiPublisher.publish(Kitti_list)
    print("Total time", time.time() - start_time_callback)

  def listener(self):

    rospy.init_node('Tracking', anonymous=True)
    rospy.Subscriber("/pp/detection", Object_kitti_list, self.callback)
    self.markerPublisher = rospy.Publisher("/AB3DMOT_markers", MarkerArray, queue_size = 10)
    self.kittiPublisher = rospy.Publisher("/AB3DMOT_kitti", Object_kitti_list, queue_size = 10)
    rospy.spin()

  def yaw2quaternion(self, yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)

  def tracking_to_rviz(self, trackers, msg):

    MarkerArray_list = MarkerArray()		##CREO EL MENSAJE GENERAL
    obj = Marker()
    obj.action = 3
    MarkerArray_list.markers.append(obj)
    self.markerPublisher.publish(MarkerArray_list)

    if trackers.size != 0:
      for d in trackers:              #d = x, y, z, theta, l, w, h, ID, other info, confidence
        if d[14] > self.threshold:
          obj = Marker()
          obj.header.stamp = rospy.Time.now()
          obj.header.frame_id = msg.header.frame_id
          obj.type = Marker.CUBE
          obj.id = int(d[7])
          obj.lifetime = rospy.Duration.from_sec(1)
          obj.pose.position.x = float(d[3])
          obj.pose.position.y = float(d[4])
          obj.pose.position.z = float(d[5])
          q = self.yaw2quaternion(float(d[6]))
          obj.pose.orientation.x = q[1] 
          obj.pose.orientation.y = q[2]
          obj.pose.orientation.z = q[3]
          obj.pose.orientation.w = q[0]
          obj.scale.x = float(d[0])
          obj.scale.y = float(d[1])
          obj.scale.z = float(d[2])
          id = int(d[7])
          obj.color.r = self.colours[id][0]  #Colour by id
          obj.color.g = self.colours[id][1]
          obj.color.b = self.colours[id][2]
          obj.color.a = 0.5
      
          MarkerArray_list.markers.append(obj)

    return MarkerArray_list

  def tracking_to_visualizer(self, trackers, msg):

    Kitti_list = Object_kitti_list()		##CREO EL MENSAJE GENERAL
    Kitti_list.header.stamp = msg.header.stamp

    if trackers.size != 0:
      for d in trackers:              #d = x, y, z, theta, l, w, h, ID, other info, confidence
        if d[14] > self.threshold:
          obj = Object_kitti()          
          obj.type.data = self.det_id2str[float(d[9])]
          obj.id = int(d[7])
          obj.loc = [float(d[3]), float(d[4]), float(d[5])]
          obj.rot = float(d[6])
          obj.dims = [float(d[0]), float(d[1]), float(d[2])]
          obj.bbox = [float(d[10]), float(d[11]), float(d[12]), float(d[13])]
          obj.score = float(d[14])
      
          Kitti_list.object_list.append(obj)

    return Kitti_list



    
if __name__ == '__main__':

  rosT = rosTracker()
  rosT.threshold = 0.6
  rosT.total_time = 0.0
  rosT.total_frames = 0
  rosT.colours = np.random.rand(int(1e6),3)
  rosT.det_id2str = {1:'Pedestrian', 2:'Car', 3:'Cyclist', 4:'Van', 5:'Truck'}
  rosT.mot_tracker = AB3DMOT() 
  rosT.listener()
