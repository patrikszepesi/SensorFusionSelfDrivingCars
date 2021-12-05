def pcl_to_bev(lidar_pcl, configs, vis=True):

    # compute bev-map discretization by dividing x-range by the bev-image height,how many meters corresponds to how many pixels in BEV space, divide lower boundary of x by the upper boundary and divide by height, this helps convert from metrics space to BEV space
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    lidar_pcl_cpy = np.copy(lidar_pcl)#create copy to not change input point cloud
    #[:,0]#get all rows from column zero which are the x values
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))#from x coordinates to BEV coordinates
    #find BEV coordinate of each lidar point, we use_int to get 8 bit integers

    # transform all metrix y-coordinates as well but center the foward-facing x-axis on the middle of the image
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)#add this term to get x axis on the middle of the image

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl_cpy[:, 2] = lidar_pcl_cpy[:, 2] - configs.lim_z[0]  # when values fall below position zero they flip, especially at road curvatures, so you have a really dark value then its so dark that it goes below zero and ends up becoming 255 and really bright, avoid thise by subtracting the lower limit of the x coordinate

    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
    #look for points that fall into the same grid cell, and make them into a single point?
    idx_height = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_hei = lidar_pcl_cpy[idx_height]

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    #[:,0:2]removes all x and ys because lower is inclusive upper is exclusive
    _, idx_height_unique = np.unique(lidar_pcl_hei[:, 0:2], axis=0, return_index=True)
    lidar_pcl_hei = lidar_pcl_hei[idx_height_unique]

    # assign the height value of each unique entry in lidar_top_pcl to the height map and
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    #normalize between 0 and 255, to this 8 bit space
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    height_map[np.int_(lidar_pcl_hei[:, 0]), np.int_(lidar_pcl_hei[:, 1])] = lidar_pcl_hei[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))

    # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity
    lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]

    # only keep one point per grid cell
    _, indices = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_pcl_int = lidar_pcl_cpy[indices]

    # create the intensity map
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 3] / (np.amax(lidar_pcl_int[:, 3])-np.amin(lidar_pcl_int[:, 3]))

    # visualize intensity map
    if vis:
        img_intensity = intensity_map * 256
        img_intensity = img_intensity.astype(np.uint8)
        while (1):
            cv2.imshow('img_intensity', img_intensity)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
