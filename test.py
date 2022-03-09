# viewing_ROIs_binary(556, 95)
#
# heatmap_binary(556, 108, 0.7)
# #
# contours_c, hierarchy_c = detect_edge(556, 108)
# polygons = contour_to_poly(contours_c, hierarchy_c)
# save_ROIs_poly(556, 108, "cumulative_time", polygons)
#
# #
# # image_path = "images/x2.5/MP_" + str(108).zfill(4) + "_x2.5_z0.tif"
# #
# # idx = 0
# for p in polygons:
#     print(p)
# #     image_path = draw_rectangle(image_path, str(556) + "_" + str(108) + "/test/", str(idx), p)
# #     idx = idx + 1
#
# poly = pd.read_csv("csvFiles/pathologists_soft_rois_poly.csv")
#
#
# roi_poly = segment_str_to_poly(poly["ROI_coords"])
#
# print("extracted:")
# for p in roi_poly:
#     print(p)

# visualize_overlapping_ROIs()
import pandas as pd

test2 = [[7,7,7,7,0,0,0,0,0,8],
         [9,2,2,9,9,0,8,8,0,8],
         [9,2,2,3,3,3,8,8,7,7],
         [0,7,7,7,0,0,8,8,0,0],
         [0,7,7,7,5,5,8,0,0,0],
         [0,7,7,7,5,5,8,0,0,0],
         [9,6,6,6,0,0,0,8,8,0],
         [9,9,9,6,6,6,8,8,0,0],
         [0,9,8,8,0,0,0,8,0,0],
         [0,9,8,8,0,0,0,0,0,0]]

test1 = [[10,10,10,7,0,0,0,0,0,8],
         [9.5,9.4,2,9.1,9.1,0,8.1,8.2,0,8],
         [9.1,9.3,2,3,3,3,8.3,8.3,7,7],
         [0,7,7,7,0,0,8,8,0,0],
         [0,7,7,7,5,5,8,0,0,0],
         [0,7,7,7,5,5,8,0,0,0],
         [9,6,6,6,0,0,0,8,8,0],
         [9,9.1,9.1,6,6,6,8,8,0,0],
         [0,9.1,8.2,8.2,0,0,0,8,0,0],
         [0,9.1,8.2,8.2,0,0,0,0,0,0]]

test3 = []

for i in range(10):
    temp = []
    for j in range(10):
        temp.append(test2[i][j] * 0.1)
    test3.append(temp)

test4 = []

for i in range(10):
    temp = []
    for j in range(10):
        if test2[i][j] >= 7:
            temp.append(1)
        else:
            temp.append(0)
    test4.append(temp)


hist = pd.read_csv("csvFiles/interpretations_histology.csv")

hist_1 = hist[(hist["true_diagnosis"] == 5)][["pathologist_id","case_id","accuracy"]]
# hist_0 = hist[(hist["true_diagnosis"] == 5)][["pathologist_id","case_id","accuracy"]]

for idx, h in hist_1.iterrows():
    print(h["pathologist_id"], h["case_id"], h["accuracy"])
# print(hist_0)