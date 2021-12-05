#visualize the intensity of the  laser reflection that arrives back at the lidar receiver,
#here we visualize the intensity of the range image, so the vehichles in front of you show up in grey scale image

def vis_intensity_channel(frame, lidar_name):

    # extract range image from frame
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    ri[ri<0]=0.0

    # map value range to 8bit
    #extract the first channel and scale it in a way that entire value range from 0.0m to 75m
    #is properly mapped to the 8bit color depth of a grayscale image
    ri_intensity = ri[:,:,1]
    ri_intensity = np.amax(ri_intensity)/2 * ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
    img_intensity = ri_intensity.astype(np.uint8)

    # focus on +/- 45° around the image center
    #In order to focus on the direction of driving, we can narrow the horizontal field-of-view to \pm 45\degree±45°
    #around the forward-facing x-axis. As stated in the Waymo dataset paper , the center of the image corresponds
    #to the positive x-axis. Also, we know that the distance between center and left as well as the distance between
    # center and right correspond to 180°. Therefore, as 45° amounts to 1/8th of the number of range image columns,
    #we can extract the center 90° with the following code:
    deg45 = int(img_intensity.shape[1] / 8)
    ri_center = int(img_intensity.shape[1]/2)
    img_intensity = img_intensity[:,ri_center-deg45:ri_center+deg45]

    cv2.imshow('intensity image', img_intensity)
    cv2.waitKey(0)


#before this line of code:
ri_intensity = np.amax(ri_intensity)/2 * ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
#Note that there is a single very bright pixel left of the center while the rest of the intensity map is mostly zero.
# The reason for this behavior of the scaling process is that the range of values extends over several powers of ten from
# the darkest to the brightest region. This is a common problem with Lidar data in an automotive context due to the presence
# of retro-reflective materials (e.g. some traffic signs, tail-lights, some license plates) in a typical scene. For such materials,
# the intensity of the reflected laser beam is significantly higher than for other materials. Therefore, if we were to normalize
# the data using standard approaches from the literature such as z-normalization or similar methods, we would succeed in mitigating the influence of
# "intensity outliers" but at the same time boost the noise level significantly.
# Therefore, a somewhat heuristic approach to this lidar-specific problem could be to
# simply multiply the entire intensity image with half the value of the max. intensity value.
# In computer vision, this operation would be termed "contrast adjustment". In code, this looks like the following:

ri_intensity = np.amax(ri_intensity)/2 * ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))


#then from range images we attempt to reconstruct a 3d thing, from which we can get a point cloud
