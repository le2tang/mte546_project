run {roslaunch realsense2_camera rs_camera.launch align_depth:=true filters:=pointcloud}
run {roslaunch body_est body_est.launch}
Need to install apriltag_ros package and maybe the apriltag and apriltag_ros git repos
