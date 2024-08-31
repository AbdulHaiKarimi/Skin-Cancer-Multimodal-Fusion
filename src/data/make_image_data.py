import glob
import os
import shutil

import pandas as pd
import tqdm

try:
    # number of patients from meta data
    meta_data = pd.read_csv("./Dataset/HAM10000_metadata.csv")

    # # total number of patients from images samples
    p1 = len(os.listdir("./Dataset/HAM10000_images_part_1"))
    p2 = len(os.listdir("./Dataset/HAM10000_images_part_2"))
    print(f"Total Number of Samples Before Preprocessing: {p1+p2}")

    p1 = os.listdir("./Dataset/HAM10000_images_part_1")
    p2 = os.listdir("./Dataset/HAM10000_images_part_2")
    p1.extend(p2)
    all_files_path = p1

    # classifying each samples correctly from the meta data
    p1_path = "./Dataset/HAM10000_images_part_1"
    p2_path = "./Dataset/HAM10000_images_part_2"
    onlyfiles = [
        f
        for f in all_files_path
        if os.path.isfile(os.path.join(p1_path, f))
        or os.path.isfile(os.path.join(p2_path, f))
    ]

    # checking files existance
    print(f"Total Number of existed sample of images: {len(onlyfiles)}")

    # preparing dataset in a proper format for keras data loaders
    if os.path.exists("HAM_10000_Dataset"):
        shutil.rmtree(
            "HAM_10000_Dataset"
        )  # remove the folder for dataset if it already exists
        os.makedirs("HAM_10000_Dataset", exist_ok=True)
    else:
        os.makedirs("HAM_10000_Dataset", exist_ok=True)

    # iterat on each sample and correctly classify it according its id in meta data
    for dx in tqdm.tqdm(meta_data["dx"].unique(), desc="In Pregress", ascii=True):
        print(f"{dx} class files in progress")

        filterd_data = meta_data[meta_data["dx"] == dx]

        if dx not in os.listdir("HAM_10000_Dataset"):
            os.makedirs(f"HAM_10000_Dataset/{dx}")

            [
                shutil.copy(os.path.join(p1_path, f), f"HAM_10000_Dataset/{dx}/")
                for f in filterd_data["image_id"] + ".jpg"
                if os.path.isfile(os.path.join(p1_path, f))
            ]
            [
                shutil.copy(os.path.join(p2_path, f), f"HAM_10000_Dataset/{dx}/")
                for f in filterd_data["image_id"] + ".jpg"
                if os.path.isfile(os.path.join(p2_path, f))
            ]

        else:
            [
                shutil.copy(os.path.join(p1_path, f), f"HAM_10000_Dataset/{dx}/")
                for f in filterd_data["image_id"] + ".jpg"
                if os.path.isfile(os.path.join(p1_path, f))
            ]
            [
                shutil.copy(os.path.join(p2_path, f), f"HAM_10000_Dataset/{dx}/")
                for f in filterd_data["image_id"] + ".jpg"
                if os.path.isfile(os.path.join(p2_path, f))
            ]
    print("Successfull Created Dataset")

    print(
        f"Number of samples in total after creating dataset: {len(glob.glob('HAM_10000_Dataset/*/*'))}"
    )

except Exception as e:
    print(e)
