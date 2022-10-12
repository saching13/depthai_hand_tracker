#!/usr/bin/env python3


# from HandTrackerRenderer import HandTrackerRenderer
from depthai_ros_msgs.msg import HandLandmarkArray, HandLandmark
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import rospy

parser = argparse.ArgumentParser()

parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument('--no_lm', action="store_true", 
                    help="Only the palm detection model is run (no hand landmark model)")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Landmark model 'full', 'lite', 'sparse' or path to a blob file")
parser_tracker.add_argument('--use_world_landmarks', action="store_true", 
                    help="Fetch landmark 3D coordinates in meter")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Solo mode: detect one hand max. If not used, detect 2 hands max (Duo mode)")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                    help="Enable spatial location measure of palm centers")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
                    help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels") 
parser_tracker.add_argument("-bpf", "--body_pre_focusing", default='higher', choices=['right', 'left', 'group', 'higher'],
                    help="Enable Body Pre Focusing")      
parser_tracker.add_argument('-ah', '--all_hands', action="store_true", 
                    help="In Body Pre Focusing mode, consider all hands (not only the hands up)")                                     
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")
parser_tracker.add_argument('--dont_force_same_image', action="store_true",
                    help="(Edge Duo mode only) Don't force the use the same image when inferring the landmarks of the 2 hands (slower but skeleton less shifted)")
parser_tracker.add_argument('-lmt', '--lm_nb_threads', type=int, choices=[1,2], default=2, 
                    help="Number of the landmark model inference threads (default=%(default)i)")  
parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, 
                    help="Print some debug infos. The type of info depends on the optional argument.")                
parser_renderer = parser.add_argument_group("Renderer arguments") 
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
parser_renderer.add_argument('-fid', '--frame_id', 
                    help="Enter the TF frame id")
args = parser.parse_args()
dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

from HandTrackerBpfEdge import HandTrackerBpf

tracker = HandTrackerBpf(
        input_src=args.input, 
        use_lm= not args.no_lm, 
        use_world_landmarks=args.use_world_landmarks,
        use_gesture=True,
        xyz=args.xyz,
        solo=args.solo,
        crop=args.crop,
        resolution=args.resolution,
        body_pre_focusing=args.body_pre_focusing,
        hands_up_only=not args.all_hands,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=args.lm_nb_threads,
        stats=True,
        trace=args.trace,
        **tracker_args
        )


frame_id = args.frame_id
# renderer = HandTrackerRenderer(
#         tracker=tracker,
#         output=args.output)

landmarkPub = rospy.Publisher('handCoordinates', HandLandmarkArray, queue_size=10)
imagePub = rospy.Publisher('rgb', Image, queue_size=10)
rospy.init_node('hand_tracker', anonymous=True)
rate = rospy.Rate(30) # 30hz
bridge = CvBridge()
seq = 0
while not rospy.is_shutdown():
    # Run hand tracker on next frame
    # 'bag' is information common to the frame and to the hands 
    # (like body keypoints in Body Pre Focusing mode)
    frame, hands, bag = tracker.next_frame()
    image_message = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
    msg = HandLandmarkArray()
    seq += 1
     
    for hand in hands:
        local_msg = HandLandmark()
        local_msg.label = hand.label
        local_msg.lm_score = hand.lm_score
        if hand.gesture != None:
            local_msg.gesture = hand.gesture
        else:
            local_msg.gesture = ''

        for x, y in hand.landmarks:
            loc = Pose2D()
            loc.x = x
            loc.y = y
            local_msg.landmark.append(loc)
        if args.xyz:
            x, y, z = hand.xyz
            local_msg.is_spatial = True
            local_msg.position.x = x / 1000
            local_msg.position.y = y / 1000
            local_msg.position.z = z / 1000
        else:
            local_msg.is_spatial = False
        msg.landmarks.append(local_msg)
        if hand.gesture == 'FIST':
            fistFound = True
    # print(hands)
    # print(msg)
    # print(frame.shape)
    # msg.header.seq = seq
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()

    landmarkPub.publish(msg)
    imagePub.publish(image_message)
    rate.sleep()

    # if frame is None: break
    # # Draw hands
    # frame = renderer.draw(frame, hands, bag)
    # key = renderer.waitKey(delay=1)
    # if key == 27 or key == ord('q'):
    #     break
# renderer.exit()
tracker.exit()
