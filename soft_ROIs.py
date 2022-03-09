import operator
import os
from collections import defaultdict

import imageio as imageio
import numpy as np
import pandas as pd
from scipy import ndimage

from helper import draw_rectangle, create_polygon_int, save_soft_rois, normalize_2d, load_heatmap, segment_str_to_poly, \
    heatmap_binary, detect_edge, viewing_ROIs_binary, contour_to_poly, save_ROIs_poly
from scipy.io import savemat
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def extract_zoompeaks(viewports):

    zoom_df = viewports["zoomLevel"]

    index = zoom_df.index

    zoom_peaks = []

    idx = 1

    for i, zoom in zoom_df.items():

        if idx > index.__len__() - 2:
            break

        if zoom_df.iloc[idx] > zoom_df.iloc[idx - 1] and zoom_df.iloc[idx + 1] < zoom_df.iloc[idx]:
            zoom_peaks.append(viewports.iloc[idx,3:7])
        idx = idx + 1


    return zoom_peaks





def extract_fixations(viewports):

    duration_df = viewports["duration"]
    zoom_df = viewports["zoomLevel"]

    index = duration_df.index

    duration = []

    idx = 1

    for i, value in duration_df.items():
        if idx == index.__len__():
            break
        # print(idx)
        # print(duration_df.iloc[idx])
        if duration_df.iloc[idx] > 2 and zoom_df.iloc[idx] > 5:
            duration.append(viewports.iloc[idx,3:7])
        idx = idx + 1
    # print(duration)

    return duration



def generate_heatmaps(viewports, output_path):

    metadata_df = pd.read_csv(output_path + "metadata.csv")

    case_id = viewports.iloc[0, 1]
    participant_id = viewports.iloc[0, 0]

    # case_id = 203
    # participant_id = 810

    # viewports_df = pd.read_csv("csvFiles/pathologists_viewport.csv")
    # viewports = viewports_df[(viewports_df["case_id"] == case_id) &
    #                          (viewports_df["pathologist_id"] == participant_id)]


    width = metadata_df[(metadata_df["case_id"] == case_id)]["width"].values[0]
    height = metadata_df[(metadata_df["case_id"] == case_id)]["height"].values[0]
    print(width, height)
    heatmap = np.zeros(shape=(height//16, width//16))
    print(heatmap.shape)

    for idx, viewport in viewports.iterrows():

        x = viewport["xPos"]//16
        y = viewport["yPos"]//16
        w = viewport["width"]//16
        h = viewport["height"]//16
        heatmap[y:y+h, x:x+w] = heatmap[y:y+h, x:x+w] + viewport["duration"]

    norm_heatmap_resized = normalize_2d(heatmap)

    savemat('heatmaps/' + str(participant_id) + '_' + str(case_id) + '_heatmap.mat', {'heatmap': norm_heatmap_resized})

    heatmap_dir = "images/heatmap/" + str(participant_id) + '_' + str(case_id) + "/"

    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)

    imageio.imwrite(heatmap_dir + "heatmap_grey.jpeg", norm_heatmap_resized)
    im_gray = cv2.imread(heatmap_dir + "heatmap_grey.jpeg", cv2.IMREAD_GRAYSCALE)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_dir + "heatmap_jet.jpeg", im_color)




def extract_cumulative_time(p_id, c_id):
# def extract_cumulative_time(heatmap):

    heatmap = load_heatmap("heatmaps/" + str(p_id) + "_" + str(c_id) + "_heatmap.mat")

    soft_rois_num = 200
    threshold = 7

    soft_rois = {}

    points_in_threshold = {}

    # Find a list of points with a value higher than the threshold and sort them descending
    for i in range(len(heatmap)):
        for j in range(len(heatmap[0])):

            if heatmap[i][j] >= threshold*0.1:

                points_in_threshold[(i, j)] = heatmap[i][j]


    heatmap_values = [1.0, 0.8]
    print(heatmap_values)



    for value in heatmap_values:
        print("heatmap value: ", value)

        if soft_rois_num > 0:
            # print(soft_rois_num, " ROIs left")
            same_value_points = [p for p, v in points_in_threshold.items() if v <= value and v > value-0.2]
            # print("points with same value: ", same_value_points)
            if len(same_value_points) == 0:
                continue

            else:

                x_min = min(p[0] for p in same_value_points)
                x_max = max(p[0] for p in same_value_points)

                y_min = min(p[1] for p in same_value_points)
                y_max = max(p[1] for p in same_value_points)

                # print("bounding box: ", x_min, x_max, y_min, y_max)
                for i in range(x_min,x_max + 1):
                    for j in range(y_min, y_max + 1):
                        # print("At: ", (i,j))
                        if len(same_value_points) > 0:
                            # print(len(same_value_points), " points left")
                            if (i, j) in same_value_points:
                                # print("start search at: ", (i,j))
                                # print("start search for width!")
                                for w in range(j, y_max + 1):
                                    # print("At: ", (i, w))
                                    if heatmap[i][w] <= value and heatmap[i][w] > value-0.2:
                                        if (i, w) in same_value_points:
                                            same_value_points.remove((i, w))
                                            # heatmap_values.remove(value)
                                            # print("removed: ", (i, w))
                                        width = 1
                                    else:
                                        width = 0
                                        break

                                width = width + w - j

                                found = False
                                # print("found width: ", width)
                                # print("start search for height!")

                                for h in range(i+1, x_max + 1):
                                    stack = []
                                    # print("h: ", h)
                                    for w in range(j, j + width):
                                        # print("w: ", w)
                                        # print("At: ", (h, w))
                                        # print((h, w))
                                        if heatmap[h][w] <= value and heatmap[h][w] > value-0.2:
                                            if (h, w) in same_value_points:
                                                same_value_points.remove((h, w))
                                                # heatmap_values.remove(value)
                                                # print("removed: ", (h, w))
                                                stack.append((h, w))
                                            # continue
                                            height = 1
                                        else:
                                            # print("seacrh interupted, points pushed back: ", stack)
                                            same_value_points.extend(stack)
                                            found = True
                                            height = 0
                                            break
                                    if found:
                                        break
                                if i == x_max:
                                    # print("height == 1")
                                    height = 1
                                else:
                                    height = height + h - i
                                # print("found height: ", height)
                                # print("soft ROI found: ", i, j, width, height, value)
                                roi_key = "roi" + str(soft_rois_num)
                                # soft_rois[roi_key] = [j, i, width, height] #xPos, yPos, width, height
                                soft_rois[roi_key] = (j*16, i*16, width*16, height*16) #xPos, yPos, width, height
                                soft_rois_num = soft_rois_num - 1
                            # else:
                            #     print("here")
                            #     break

                        else:
                            # print("Done with points with value: ", value)
                            break

                # break
        else:
            # print("Done searching")
            break

    soft_rois_df = pd.DataFrame.from_dict(soft_rois, orient='index', columns=['xPos', 'yPos', 'width', 'height'])

    return soft_rois_df


def extract_slowpanning(viewports):

    index = viewports.index

    zoom_threshold = 5
    panning_threshold = 100
    duration_threshold = 1

    slow_panning_idx = []
    idx_so_far = []
    duration_so_far = 0

    old_zoom = viewports["zoomLevel"].iloc[0]

    idx = 1

    for i, value in viewports.items():
        if idx == index.__len__():
            break

        current_zoom = viewports["zoomLevel"].iloc[idx]

        if current_zoom > zoom_threshold:
            if current_zoom == old_zoom:
                if viewports["displacement"].iloc[idx] < (panning_threshold * current_zoom):
                    duration_so_far = duration_so_far + viewports["duration"].iloc[idx]
                    if len(idx_so_far) == 0:
                        idx_so_far = [idx - 1]

                    idx_so_far.append(idx)
                else:
                    if duration_so_far > duration_threshold:
                        slow_panning_idx.extend(idx_so_far)

                    idx_so_far = []
                    duration_so_far = 0
            else:
                old_zoom = current_zoom
                if duration_so_far > duration_threshold:
                    slow_panning_idx.extend(idx_so_far)

                idx_so_far = [];
                duration_so_far = 0;

        idx = idx + 1


    slow_panning = []

    for i in slow_panning_idx:

        slow_panning.append(viewports.iloc[i,3:7])

    # print(slow_panning)


    return slow_panning


def extract_soft_rois(participants, output_path):

    interpretations_df = pd.read_csv(output_path + participants + "_interpretations.csv")
    viewports_df = pd.read_csv(output_path + participants + "_viewport.csv")

    class_5 = pd.read_csv(output_path + "class5.csv")
    class_5_list = list(class_5["class5"])

    for idx, interpretation in interpretations_df.iterrows():
        if interpretation["case_id"] in class_5_list:
            print(interpretation["pathologist_id"], interpretation["case_id"])
            interpretation_viewports = viewports_df[(viewports_df["pathologist_id"] == interpretation["pathologist_id"]) &
                                                (viewports_df["case_id"] == interpretation["case_id"])]

            generate_heatmaps(interpretation_viewports, output_path)
            # cumulative_time = extract_cumulative_time(interpretation["pathologist_id"], interpretation["case_id"])
            # save_soft_rois(participants, output_path, cumulative_time, interpretation["case_id"], interpretation["pathologist_id"], "cumulative_time")


            # zoom_peaks = extract_zoompeaks(interpretation_viewports)
            # save_soft_rois(participants, output_path, zoom_peaks, interpretation["case_id"], interpretation["pathologist_id"], "zoom_peak")
            #
            #
            # fixations = extract_fixations(interpretation_viewports)
            # save_soft_rois(participants, output_path, fixations, interpretation["case_id"], interpretation["pathologist_id"], "fixations_5")
            #
            # slow_pannings = extract_slowpanning(interpretation_viewports)
            # save_soft_rois(participants, output_path, slow_pannings, interpretation["case_id"], interpretation["pathologist_id"], "slow_panning")


# extract_soft_rois("pathologists", "csvFiles/")

def visualize_soft_rois(participants, output_path, sroi, p_id, c_id):

    soft_rois_df = pd.read_csv(output_path + participants + "_soft_rois_class5_bg_removed.csv")

    soft_rois_interpretation = soft_rois_df[(soft_rois_df["pathologist_id"] == p_id) &
                                            (soft_rois_df["case_id"] == c_id)]
                                            # & (soft_rois_df["type"] == type)]
    # soft_rois_interpretation = sroi[(sroi["type"] == "cumulative_time")]


    # soft_rois_case = soft_rois[925][61]

    image_path = "images/x2.5/MP_" + str(c_id).zfill(4) + "_x2.5_z0.tif"
    # print(image_path)
    for idx, rect in soft_rois_interpretation.iterrows():
        # print(rect["xPos"]/16, rect["yPos"]/16, rect["width"]/16, rect["height"]/16)
        poly = create_polygon_int(rect["xPos"]/16, rect["yPos"]/16, rect["width"]/16, rect["height"]/16)

        image_path = draw_rectangle(image_path, str(p_id) + "_" + str(c_id) + "/test/", rect["type"] + str(idx), poly)



# remove ROIs that have background
def remove_BG_ROI():

    segments_df = pd.read_csv("csvFiles/segments.csv")

    soft_rois_df1 = pd.read_csv("csvFiles/pathologists_soft_rois.csv")
    # soft_rois_df1 = soft_rois_df[(soft_rois_df["case_id"] == case_id) & (soft_rois_df["pathologist_id"] == p_id)]

    for idx, roi in soft_rois_df1.iterrows():

        roi_poly = create_polygon_int(roi["xPos"], roi["yPos"], roi["width"], roi["height"])

        segments = segments_df[(segments_df["Case_id"] == roi["case_id"])]["segment_coords"]

        segments_poly = segment_str_to_poly(segments)
        intersect_flag = 0
        for segment in segments_poly:

            if roi_poly.intersection(segment).area/roi_poly.area >= 0.1:
                intersect_flag = 1
                break


        if intersect_flag == 0:
            soft_rois_df1 = soft_rois_df1.drop(index=idx)

    soft_rois_df1.to_csv("csvFiles/pathologists_soft_rois_bg_removed.csv", index=False)

    return soft_rois_df1


def visualize_overlapping_ROIs():

    class_5 = pd.read_csv("csvFiles/meeting.csv")
    class_5_list = list(class_5["class5"])

    interpretations_df = pd.read_csv("csvFiles/pathologists_interpretations.csv")

    for idx, interpretation in interpretations_df.iterrows():

        case_id = interpretation["case_id"]
        pathologist_id = interpretation["pathologist_id"]

        if case_id in class_5_list:
            viewing_ROIs_binary(pathologist_id, case_id)

            heatmap_binary(pathologist_id, case_id, 0.7)

            contours_c, hierarchy_c, contours_z, hierarchy_z, contours_p, hierarchy_p, contours_f, hierarchy_f = \
                detect_edge(pathologist_id, case_id)

            if hierarchy_c is not None:
                polygons_c = contour_to_poly(contours_c, hierarchy_c)
                save_ROIs_poly(pathologist_id, case_id, "cumulative_time", polygons_c)

            if hierarchy_z is not None:
                polygons_z = contour_to_poly(contours_z, hierarchy_z)
                save_ROIs_poly(pathologist_id, case_id, "zoom_peak", polygons_z)

            if hierarchy_p is not None:
                polygons_p = contour_to_poly(contours_p, hierarchy_p)
                save_ROIs_poly(pathologist_id, case_id, "slow_panning", polygons_p)

            if hierarchy_f is not None:
                polygons_f = contour_to_poly(contours_f, hierarchy_f)
                save_ROIs_poly(pathologist_id, case_id, "fixation5", polygons_f)

            print(pathologist_id, case_id)

# visualize_overlapping_ROIs()
def ROIs_merge_pathologists():

    soft_rois_df = pd.read_csv("csvFiles/pathologists_soft_rois_bg_removed.csv")

    soft_rois_df.drop(columns=["pathologist_id"])

    soft_rois_df.to_csv("csvFiles/pathologists_soft_rois_bg_removed_merge_pathologists.csv", index=False)







