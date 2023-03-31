run {roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud}
run {roslaunch body_est body_est.launch}
Install the apriltag_ros package. may also need to clone the apriltag repo and apriltag_ros repo

