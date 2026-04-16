import pyzed.sl as sl
import numpy as np

def main():
    # Create a ZED Camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    # INSTEAD OF LIVE CAMERA, USE A RECORDED FILE
    init_params.set_from_svo_file("path_to_your_file.svo") 
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA # High quality depth
    init_params.coordinate_units = sl.UNIT.METER # Standard for UGV

    # Open the camera (SVO file)
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening SVO: {err}")
        exit()

    # Prepare containers for data
    point_cloud = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    while True:
        # Grab a new frame from the SVO
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the colored point cloud
            # This is your "LiDAR-like" data
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            # Access the raw data as a numpy array for processing
            # cloud_data[y, x] gives you [X, Y, Z, Color]
            cloud_data = point_cloud.get_data()
            
            # Example: Get distance to the center pixel
            center_point = cloud_data[cloud_data.shape[0]//2, cloud_data.shape[1]//2]
            print(f"Distance to center: {center_point[2]:.2f} meters")

        elif zed.get_svo_position() >= zed.get_svo_number_of_frames() - 1:
            print("End of SVO reached.")
            break

    zed.close()

if __name__ == "__main__":
    main()