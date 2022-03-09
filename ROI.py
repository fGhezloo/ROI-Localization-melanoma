import pandas as pd
from shapely.geometry import Polygon

from features import add_feature_to_table


def create_polygon(xpos, ypos, width, height):

    x = xpos._get_value(0, 'xPos')
    y = ypos._get_value(0, 'yPos')
    w = width._get_value(0, 'width')
    h = height._get_value(0, 'height')

    p = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])

    return p

def create_polygon_int(x, y, w, h):

    p = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)])

    return p


def contains_ROI(ROI_poly, viewport_poly):

    sizeRatio = 0.1
    intersectionRatio = 0.4

    if ROI_poly.contains(viewport_poly) or ( viewport_poly.area < ROI_poly.area and (viewport_poly.intersection(ROI_poly).area / viewport_poly.area) > intersectionRatio):
        overlay = 1

    elif (viewport_poly.intersection(ROI_poly).area / ROI_poly.area) > intersectionRatio:
        if (ROI_poly.area / viewport_poly.area) > sizeRatio :
            overlay = 1
        else:
            overlay = 0
    else:
        overlay = 0

    return overlay


def viewport_overlay_ROI(ROI_path, viewport_path, pathologist, case):


    ROI_df = pd.read_csv(ROI_path)

    ROIs = ROI_df[(ROI_df['CaseID'] == case)]
    ROI_poly = []

    for i, roi in ROIs.iterrows():
        ROI_poly.append(create_polygon_int(roi["xPos"], roi["yPos"], roi["width"], roi["height"]))


    viewports_df = pd.read_csv(viewport_path)

    interpretation_viewport = viewports_df[(viewports_df["pathologist_id"] == pathologist) & (viewports_df["case_id"] == case)]

    overlay_all = [0] * len(interpretation_viewport)

    viewport_idx = 0
    for idx, viewport in interpretation_viewport.iterrows():
        viewport_poly = create_polygon_int(viewport["xPos"], viewport["yPos"], viewport["width"], viewport["height"])

        for roi_poly in ROI_poly:
            overlay = contains_ROI(roi_poly, viewport_poly)
            if overlay == 1:
                overlay_all[viewport_idx] = 1

        viewport_idx = viewport_idx + 1

    return overlay_all


def ROI_time_feature(viewport_path, features_path, output_path):

    viewports_df = pd.read_csv(viewport_path)

    interpretations_df = pd.read_csv(output_path + "interpretations.csv")

    ROI_time_percentage = []


    for idx1, interpretation in interpretations_df.iterrows():
        overlay = viewport_overlay_ROI(output_path + "consensus_ROIs.csv", viewport_path, interpretation["pathologist_id"], interpretation["case_id"])

        interpretation_duration = viewports_df[(viewports_df["pathologist_id"] == interpretation["pathologist_id"]) & (viewports_df["case_id"] == interpretation["case_id"])]["duration"]
        duration_list = list(interpretation_duration)
        total_duration = sum(duration_list)
        ROIduration_sum = 0
        ROI_viewed = 0

        for idx2, o in enumerate(overlay):
            if o == 1:
                ROI_viewed += 1
                ROIduration_sum += duration_list[idx2]

        ROI_time_ratio = 100 * ROIduration_sum / total_duration
        ROI_time_percentage.append("{:.2f}".format(ROI_time_ratio))

    add_feature_to_table(features_path, "ROI_time_percentage", ROI_time_percentage)