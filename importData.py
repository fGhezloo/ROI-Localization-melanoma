import csv

import numpy as np
import pandas as pd
from scipy.io import loadmat

import helper

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',20)

# import raw tracking data
def import_tracking_file(participants, input_path, output_path):

    df = pd.read_fwf(input_path + participants + "_viewport.rpt", delimiter = ' ')

    pathologistID = []
    caseID = []
    time = list(df['LogTime'])[1:-1]
    xPos = list(df['PositionX'])[1:-1]
    yPos = list(df['PositionY'])[1:-1]
    width = list(df['Width'])[1:-1]
    height = list(df['Height'])[1:-1]
    zoomLevel = list(df['ZoomLevel'])[1:-1]


    for index, row in df.iterrows():
        if index == 0 or index == len(df.values)-1:
            continue
        p_c_string = row['ViewportLogId ParticipantId CaseId'].split()
        pathologistID.append(p_c_string[1])
        caseID.append(p_c_string[2])


    viewport_df = pd.DataFrame({'pathologist_id': pathologistID,
                                'case_id': caseID,
                                'time': time,
                                'xPos': xPos,
                                'yPos': yPos,
                                'width': width,
                                'height': height,
                                'zoomLevel': zoomLevel,
                                })
    viewport_df.to_csv(output_path + participants +"_viewport.csv", index=False)

# Read the raw viewport tracking data and extract case_ids and pathologist_ids
# Import dimensions file and save caseID, width, and height as a metadata file
def import_case_patho(participants, input_path, output_path):

    # Case IDs
    viewport_df = pd.read_csv(output_path + participants + "_viewport.csv")

    c_ids = list(viewport_df['case_id'].unique())

    case_df = pd.DataFrame(c_ids, columns =['case_id'])

    case_df.to_csv(output_path + participants + "_case_ids.csv", index=False)

    # Pathologists IDS
    p_ids = list(viewport_df['pathologist_id'].unique())

    patho_df = pd.DataFrame(p_ids, columns =['pathologist_id'])

    patho_df.to_csv(output_path + participants + "_ids.csv", index=False)


# import case sizes
def import_metadata(input_path, output_path):

    # metadata_mat = loadmat(input_path + "metadata.mat")
    #
    # fileInfo = metadata_mat["fileSizes"]
    #
    # cIds = [row[0] for row in fileInfo]
    # width = [row[2] for row in fileInfo]
    # height = [row[1] for row in fileInfo]
    #
    # metadata_df = pd.DataFrame({'CaseID': cIds,
    #                             'Width': width,
    #                             'Height': height})
    #
    # metadata_df.to_csv(output_path + "metadata.csv", index=False)

    metadata_mat = loadmat(input_path + "metadata.mat")
    #
    # case_ids = metadata_mat["case_ids"]
    # with open(output_path + 'case_ids.csv','w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Case_id'])
    #     writer.writerows(case_ids)
    #
    # pathologist_ids = metadata_mat["expert_ids"]
    # with open(output_path + 'expert_ids.csv','w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['pathologist_id'])
    #     writer.writerows(pathologist_ids)


    fileInfo = metadata_mat["fileSizes"]
    cIds = [row[0] for row in fileInfo]
    width = [row[2] for row in fileInfo]
    height = [row[1] for row in fileInfo]

    metadata_df = pd.DataFrame({'case_id': cIds,
                            'width': width,
                            'height': height})

    metadata_df.to_csv(output_path + "metadata.csv", index=False)


# Remove viewports entries longer than 1 minute and out of image range
def clean_tracking_data(participants, output_path):


    df = pd.read_csv(output_path + participants + "_case_ids.csv")
    df.drop(df[df['case_id'] == 90].index, inplace=True)
    df.drop(df[df['case_id'] == 107].index, inplace=True)
    caseIds = [x[0] for x in df.values]

    df.to_csv(output_path + participants + "_case_ids.csv", index=False)


    df = pd.read_csv(output_path + participants + "_ids.csv")
    pathologistIds = [x[0] for x in df.values]


    viewport_df = pd.read_csv(output_path + participants + "_viewport.csv")

    metadata_df = pd.read_csv(output_path + "metadata.csv")

    interpretations = pd.DataFrame()
    durations = []
    displacements = []

    for cid in caseIds:
        for pid in pathologistIds:
            # print(cid)
            image_w = metadata_df[metadata_df["case_id"] == cid]["width"].values
            image_h = metadata_df[metadata_df["case_id"] == cid]["height"].values
            # print(image_h, image_w)
            # Check for out of image range viewports
            interpretation = viewport_df[(viewport_df["case_id"] == cid) & (viewport_df["pathologist_id"] == pid)\
                                         & (viewport_df["xPos"] > 0) & (viewport_df["yPos"] > 0)\
                                         & ((viewport_df["xPos"] + viewport_df["width"]) <= image_w[0])\
                                         & ((viewport_df["yPos"] + viewport_df["height"]) <= image_h[0])]

            if(interpretation.empty):
                continue

            time = interpretation["time"]
            duration = helper.calculate_duration(time[:-1], time[1:])
            durations.extend(duration)

            viewport = interpretation[["xPos", "yPos", "width", "height"]]
            displacement = helper.calculate_displacement(viewport)
            displacements.extend(displacement)

            interpretations = interpretations.append(interpretation)


    interpretations["duration"] = durations
    interpretations["displacement"] = displacements


    # Check for duration more than 1 minutes
    interpretations.drop(interpretations[interpretations['duration'] > 60].index, inplace=True)

    interpretations.to_csv(output_path + participants + "_viewport.csv", index=False)


# def add_displacement(participants, output_path):
#
#     viewports_df = pd.read_csv(output_path + participants + "_viewport.csv")
#     interpretations_df = pd.read_csv(output_path + participants + "_interpretations.csv")
#
#     for idx, interpretation in interpretations_df.iterrows():
#         interpretation_viewports = viewports_df[(viewports_df["pathologist_id"] == interpretation["pathologist_id"]) &
#                                                 (viewports_df["case_id"] == interpretation["case_id"])]






def import_ROI(input_path, output_path):

    # df_roi = pd.read_excel(ROI_path)
    #
    # rois_reorder = df_roi[['CaseID', 'xPos', 'yPos', 'width', 'height']]
    #
    # rois_reorder.to_csv(output_path + "consensus_ROIs.csv", index=False)

    consensus_ROI_mat = loadmat(input_path + "expert_hard_roi_and_consensus.mat")

    consensus_ROI = [i[0] for row in consensus_ROI_mat["consensus_rectangles"] for i in row]


    case_ids = pd.read_csv(output_path + "experts_case_ids.csv")
    case_list = list(case_ids["case_id"])

    xPos = []
    yPos = []
    width = []
    height = []
    print(len(case_list))
    print(len(consensus_ROI))

    for idx, ROI in enumerate(consensus_ROI):
        xPos.append(ROI[1])
        yPos.append(ROI[0])
        width.append(ROI[3])
        height.append(ROI[2])


    ROI_df = pd.DataFrame({'case_id': case_list,
                           'xPos': xPos,
                           'yPos': yPos,
                           'width': width,
                           'height': height})

    ROI_df.to_csv(output_path + "consensus_ROIs.csv", index=False)




# def import_histology_survey_data(histologyFile, surveyFile, pathologists):
#
#     pathologists_ids = pd.read_csv(pathologists)
#     plist = pathologists_ids["pathologist_id"].tolist()
#
#     histologyMat = loadmat(histologyFile)
#     surveyMat = loadmat(surveyFile)
#
#     pid = [i[0] for i in surveyMat["id"]]
#     ageCategory = [i[0] for i in surveyMat["ageCategory"]]
#     caseload = [i[0] for i in surveyMat["caseload"]]
#     gender = [i[0] for i in surveyMat["gender"]]
#     challenging = [i[0] for i in surveyMat["challenging"]]
#     confidence = [i[0] for i in surveyMat["confident"]]
#     fellowship = [i[0] for i in surveyMat["fellowship"]]
#     experience = [i[0] for i in surveyMat["years"]]
#
#     pathologists_df = pd.DataFrame(columns=['pathologist_id','gender','ageCategory','fellowship', 'experience', 'caseload', 'challenging', 'confidence'])
#
#
#     for idx, p in enumerate(pid):
#         if p in plist:
#             pathologists_df = pathologists_df.append({
#                 "pathologist_id": p,
#                 "gender":  gender[idx],
#                 "ageCategory": ageCategory[idx],
#                 "fellowship":  fellowship[idx],
#                 "experience": experience[idx],
#                 "caseload":  caseload[idx],
#                 "challenging": challenging[idx],
#                 "confidence":  confidence[idx]
#             }, ignore_index=True)
#
#     pathologists_df.to_csv("csvFiles/Pathologists_with_demographic.csv", index=False)
#
#
#     pid_h = [i[0] for i in histologyMat["participant_id"]]
#     cid = [i[0] for i in histologyMat["case_id"]]
#     diagnosis = [i[0] for i in histologyMat["diagnosis"]]
#     trueDiagnosis = [i[0] for i in histologyMat["true_diagnosis"]]
#     difficulty = [i[0] for i in histologyMat["difficulty"]]
#     confidence_h = [i[0] for i in histologyMat["confidence"]]
#
#
#     interpretation_df = pd.DataFrame(columns=['pathologist_id','case_id','diagnosis','true_diagnosis', 'accuracy', 'difficulty', 'confidence'])
#
#     for idx, p in enumerate(pid_h):
#         if p in plist:
#             interpretation_df = interpretation_df.append({
#                 "pathologist_id": p,
#                 "case_id":  cid[idx],
#                 "diagnosis": diagnosis[idx],
#                 "true_diagnosis":  trueDiagnosis[idx],
#                 "accuracy": 1 if diagnosis[idx] == trueDiagnosis[idx] else 0,
#                 "difficulty": difficulty[idx],
#                 "confidence": confidence_h[idx]
#             }, ignore_index=True)
#
#     interpretation_df.to_csv("csvFiles/interpretations_histology.csv", index=False)



# def viewports_with_histology(histology, viewports):
#
#     histology_df = pd.read_csv(histology)
#
#     viewports_df = pd.read_csv(viewports)
#
#
#     merged_df = histology_df.merge(viewports_df, how='inner', left_on=["pathologist_id", "case_id"], right_on=["pathologistID", "caseID"])
#     merged_df.drop('pathologistID', inplace=True, axis=1)
#     merged_df.drop('caseID', inplace=True, axis=1)
#     merged_df.drop('diagnosis', inplace=True, axis=1)
#     merged_df.drop('true_diagnosis', inplace=True, axis=1)
#     merged_df.drop('accuracy', inplace=True, axis=1)
#     merged_df.drop('difficulty', inplace=True, axis=1)
#     merged_df.drop('confidence', inplace=True, axis=1)
#
#
#     p_c_pairs = []
#
#     for idx, row in merged_df.iterrows():
#         if (row["pathologist_id"], row["case_id"]) not in p_c_pairs:
#             p_c_pairs.append((row["pathologist_id"], row["case_id"]))
#
#
#     droped_list = []
#     for idx, row in histology_df.iterrows():
#         if (row["pathologist_id"], row["case_id"]) not in p_c_pairs:
#             droped_list.append(idx)
#
#     updated_histology_df = histology_df.drop(droped_list)
#
#     merged_df.to_csv("csvFiles/Pathologists_ViewPort_with_histology.csv", index=False)
#     updated_histology_df.to_csv("csvFiles/interpretations_histology.csv", index=False)



