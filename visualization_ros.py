import os, numpy as np, sys, cv2
from PIL import Image
from utils import is_path_exists, mkdir_if_missing, load_list_from_folder, fileparts, random_colors
from kitti_utils import read_label, compute_box_3d, draw_projected_box3d, Calibration, project_to_image

import rospy 
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from t4ac_perception_msgs.msg import Object_kitti_list, Object_kitti
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

max_color = 30
colors = random_colors(max_color)       # Generate random colors
type_whitelist = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck']
score_threshold = -10000
width = 1920
height = 1080

class Object3d(object):

    def __init__(self, obj):

        self.type = obj.type # 'Car', 'Pedestrian', ...
        self.truncation = -1 # truncated pixel ratio [0..1]
        self.occlusion = -1 # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = -1 # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = obj.bbox[0] # left
        self.ymin = obj.bbox[1] # top
        self.xmax = obj.bbox[2] # right
        self.ymax = obj.bbox[3] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])

        # extract 3d bounding box information
        self.h = obj.dims[0] # box height
        self.w = obj.dims[1] # box width
        self.l = obj.dims[2] # box length (in meters)
        self.t = obj.loc # location (x,y,z) in camera coord.
        self.ry = obj.rot # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        self.score = obj.score

        #self.score = float(data[15])
        self.id = int(obj.id)

    def quaternion_to_euler(x, y, z, w):

            import math
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            X = math.degrees(math.atan2(t0, t1))

            t2 = +2.0 * (w * y - z * x)
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            Y = math.degrees(math.asin(t2))

            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            Z = math.degrees(math.atan2(t3, t4))

            return X, Y, Z

class Visualizer:


    def show_image_with_boxes(self, img, objects_res, calib, save_path, height_threshold=0):
        img2 = np.copy(img) 
        for obj in objects_res:
            #box3d_pts_2d = compute_box_3d(obj, calib.P, calib.V2C)
            box3d_pts_2d = compute_box_3d(obj, calib.P, calib.V2C)
            color_tmp = tuple([int(tmp * 255) for tmp in colors[obj.id % max_color]])
            #print("box3d_pts_2d", box3d_pts_2d)
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=color_tmp)
            text = 'ID: %d' % obj.id# + ', s: %.2f' % obj.score
            if box3d_pts_2d is not None:
                img2 = cv2.putText(img2, text, (int(box3d_pts_2d[4, 0]), int(box3d_pts_2d[4, 1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=color_tmp) 

        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        #cv2.imwrite(save_path, img2)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(img2, "rgb8"))

    def callback(self, infomsg, imagemsg):
        objects = []
        self.count += 1
        print("Frame: ", self.count)
        for obj in infomsg.object_list:
            obj3d = Object3d(obj)
            objects.append(obj3d)

        image = self.bridge.imgmsg_to_cv2(imagemsg, desired_encoding='passthrough')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_path = "/home/robesafe/AB3DMOT_v2/results/pcdet_KITTI_val/trk_image_vis/0001/" + str(self.count).zfill(6) + ".png"
        self.show_image_with_boxes(image, objects, self.calib, image_path)




    def listener(self):

        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        self.count = 0
        #calib_file = "/home/robesafe/AB3DMOT/data/KITTI/resources/CARLA/calib/0102.txt"
        #calib_file = "/media/robesafe/videos1.2/KITTI/KITTI_dataset_tracking/data_tracking_calib/training/calib/0001.txt"
        calib_file = "t4ac_calib.txt"
        self.calib = Calibration(calib_file)			# load the calibration
        self.bridge = CvBridge()

        rospy.init_node('visualizer', anonymous=True)

        subs_info = Subscriber("/AB3DMOT_kitti",   Object_kitti_list, queue_size = 5)
        subs_img   = Subscriber("/zed/zed_node/left/image_rect_color",      Image, queue_size = 5)
        self.image_pub = rospy.Publisher("/AB3DMOT_image", Image)
        #subs_img   = Subscriber("/carla/ego_vehicle/camera/rgb/view/image_color",      Image, queue_size = 5)
        ats = ApproximateTimeSynchronizer([subs_info, subs_img], queue_size=5, slop=10)
        ats.registerCallback(self.callback)

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()

if __name__ == '__main__':
    vis = Visualizer()
    vis.listener()

