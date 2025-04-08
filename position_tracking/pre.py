# %%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.spatial import distance as scipy_distance
from tqdm import tqdm
# %%

def find_circles_adaptive(current_frame_gray, num_of_rafts, radii_hough,
                          adaptive_thres_blocksize=9, adaptive_thres_const=-20,
                          min_sep_dist=20, raft_center_threshold=60,
                          top_left_x=390, top_left_y=450, width_x=850, height_y=850):
    """
    find the centers of each raft
    :param current_frame_gray:
    :param num_of_rafts:
    :param radii_hough:
    :param adaptive_thres_blocksize:
    :param adaptive_thres_const:
    :param min_sep_dist:
    :param raft_center_threshold:
    :param top_left_x:
    :param top_left_y:
    :param width_x:
    :param height_y:
    :return: raft_centers, raft_radii, raft_count

    """
    # key data set initialization
    raft_centers = np.zeros((num_of_rafts, 2), dtype=int)
    raft_radii = np.zeros(num_of_rafts, dtype=int)

    # crop the image
    image_cropped = current_frame_gray[top_left_y: top_left_y + height_y, top_left_x: top_left_x + width_x]

    # threshold the image
    image_thres = cv.adaptiveThreshold(image_cropped, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                       adaptive_thres_blocksize, adaptive_thres_const)
    plt.imshow(image_thres, cmap='gray')
    # use Hough transform to find circles
    hough_results = hough_circle(image_thres, np.arange(*radii_hough))
    accums, cx, cy, radii = hough_circle_peaks(hough_results, np.arange(*radii_hough))
    # assuming that the first raft (highest accumulator score) is a good one
    #    raft_centers[0,0] = cx[0]
    #    raft_centers[0,1] = cy[0]
    #    raft_radii[0] = radii[0]
    raft_count = 0  # starting from 1!

    # remove circles that belong to the same raft and circles that happened to be in between rafts
    for accumScore, detected_cx, detected_cy, detected_radius in zip(accums, cx, cy, radii):
        new_raft = 1
        if image_cropped[detected_cy, detected_cx] < raft_center_threshold:
            new_raft = 0
        elif image_cropped[detected_cy - detected_radius // 2: detected_cy + detected_radius // 2,
                           detected_cx - detected_radius // 2:detected_cx + detected_radius // 2].mean() \
                < raft_center_threshold:
            new_raft = 0
        #        elif  (detected_cx - width_x/2)**2 +  (detected_cy - height_y/2)**2 > lookup_radius**2:
        #            new_raft = 0
        else:
            cost_matrix = scipy_distance.cdist(np.array([detected_cx, detected_cy], ndmin=2),
                                               raft_centers[:raft_count, :], 'euclidean')
            if np.any(cost_matrix < min_sep_dist):  # raft still exist
                new_raft = 0
        if new_raft == 1:
            raft_centers[raft_count, 0] = detected_cx
            # note that raft_count starts with 1, also note that cx corresonds to columns number
            raft_centers[raft_count, 1] = detected_cy
            # cy is row number
            raft_radii[raft_count] = detected_radius
            raft_count = raft_count + 1
        if raft_count == num_of_rafts:
            #            error_message = 'all rafts found'
            break

    # convert the xy coordinates of the cropped image into the coordinates of the original image
    raft_centers[:, 0] = raft_centers[:, 0] + top_left_x
    raft_centers[:, 1] = raft_centers[:, 1] + top_left_y

    return raft_centers, raft_radii, raft_count

# %%
def label_images():
    detected_images = []
    detected_centers = []

    for i in tqdm(range(155)):
        try:
            image = cv.imread(f'../images/camera_image_{i:04d}.png')
            
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image =cv.resize(image, (506,506))
            mask = (image[:, :, 2] > image[:, :, 1]) & (image[:, :, 2] > image[:, :, 0])
            image[mask] = 0
            gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            centers, radii, count = find_circles_adaptive(gray_image, 7, [3, 15], top_left_x=160, top_left_y=160, width_x=200, height_y=200, adaptive_thres_blocksize=7, adaptive_thres_const=-10, raft_center_threshold=80, min_sep_dist=11)
            if count == 7:
                detected_images.append(gray_image)
                detected_centers.append(centers)
            else:
                print(f"detected {count} particles in image:{i}")
        except:
            print(f"Failed to read image {i}")
    detected_centers = np.array(detected_centers)
    detected_images = np.array(detected_images)

    detected_images =(detected_images- detected_images.mean())/detected_images.std()
    # detected_images = np.expand_dims(detected_images, axis=-1)

    print(detected_centers.shape)
    print(detected_images.shape)    
    np.save('detected_centers.npy', detected_centers)
    np.save('detected_images.npy', detected_images)

if __name__ == '__main__':
    label_images()
