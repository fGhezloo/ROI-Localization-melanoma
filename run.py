import os
import shutil


from importData import import_metadata, clean_tracking_data, import_ROI, import_case_patho, import_tracking_file
from helper import drop_feature_from_table, export_interpretations
from features import total_time_feature, zoom_level_features, magnification_feature, scanning_percentage_feature
from ROI import ROI_time_feature
from soft_ROIs import extract_soft_rois

from soft_ROIs import extract_fixations, extract_zoompeaks, extract_cumulative_time


def run(args):

    if args.task == "import":
        import_tracking_file(args.participants, args.input_path, args.output_path)
        import_case_patho(args.participants, args.input_path, args.output_path)
        import_metadata(args.input_path, args.output_path)

    elif args.task == "preprocess":
        clean_tracking_data(args.participants, args.output_path)
        export_interpretations(args.participants, args.output_path)

    elif args.task == "import_roi":
        import_ROI(args.input_path, args.output_path)

    elif args.task == "extract_soft_rois":
        extract_soft_rois(args.participants, args.output_path)

    elif args.task == "add_feature":

        if not os.path.exists(args.features_path):
            shutil.copy(args.output_path + "interpretations.csv", args.features_path)

        if args.feature == "all":
            total_time_feature(args.viewport_path, args.features_path, args.output_path)
            zoom_level_features(args.viewport_path, args.features_path, args.output_path)
            magnification_feature(args.viewport_path, args.features_path, args.output_path)
            ROI_time_feature(args.viewport_path, args.features_path, args.output_path)
            scanning_percentage_feature(args.viewport_path, args.features_path, args.output_path)

        elif args.feature == "total_time":
            total_time_feature(args.viewport_path, args.features_path, args.output_path)
        elif args.feature == "zoom_level":
            zoom_level_features(args.viewport_path, args.features_path, args.output_path)
        elif args.feature == "magnification":
            magnification_feature(args.viewport_path, args.features_path, args.output_path)
        elif args.feature == "ROI_time_percentage":
            ROI_time_feature(args.viewport_path, args.features_path, args.output_path)
        elif args.feature == "scanning_percentage":
            scanning_percentage_feature(args.viewport_path, args.features_path, args.output_path)

    elif args.task == "drop_feature":
        if args.feature == "all":
            os.remove(args.features_path)
        else:
            drop_feature_from_table(args.features_path, args.feature)

    elif args.task == "soft_rois":
        extract_zoompeaks()
        extract_fixations()
        extract_cumulative_time()
