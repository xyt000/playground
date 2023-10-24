import SimpleITK as sitk
import cv2
import numpy as np
from torchio.transforms import Affine
import pandas as pd


def getLandmarksFromTXTFile(file, split=','):
    """
    Extract each landmark point line by line from a text file, and return
    vector containing all landmarks.
    """
    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            landmarks.append([float(k) for k in line.split('\n')[0].split(split)])
        landmarks = np.asarray(landmarks).reshape((-1, 3))
        return landmarks


def getLandmarksFromExcelFile(file, scale=1.0):
    dat = pd.read_excel(file)
    return {name: (x * scale, y * scale, z * scale) for name, x, y, z in zip(dat.landmark, dat.x, dat.y, dat.z)}


if __name__ == '__main__':
    # affine transform
    rxs = []
    rys = []
    rzs = range(0, 360, 5)
    for rz in rzs:
        rotation_degrees = [0, 0, rz]
        translations = [-49, 35, 0]
        rotation_radians = np.radians(rotation_degrees).tolist()
        img_rotation = Affine(scales=[1.0, 1.0, 1.0], degrees=rotation_degrees, translation=translations,
                              center='image')
        pos_rotation = sitk.Euler3DTransform()
        pos_rotation.SetRotation(*rotation_radians)
        pos_rotation.SetTranslation(translations)

        # data
        img_path = '/home/ws/ml0077/work/rl-medical/src/data/landmarks_test2/landmarks_test/test_reference_1263_94-1_scale2_crop.tif'
        # '/home/ws/ml0077/work/rl-medical/src/data/images/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217005829488_S18402_I40731_Normalized_to_002_S_0295.nii.gz'
        landmark_path = '/home/ws/ml0077/work/rl-medical/src/data/landmarks_test2/landmarks_test/test_5_landmarks.xlsx'  # '/home/ws/ml0077/work/rl-medical/src/data/landmarks/ADNI_002_S_0816_MR_MPR__GradWarp__B1_Correction__N3__Scaled_Br_20070217005829488_S18402_I40731.txt'
        landmark_names = ['mandible dentry', 'hyoid fusion', 'first vertebra', 'optic nerve head R',
                          'optic nerve head L']
        sitk_image = sitk.ReadImage(img_path, sitk.sitkFloat32)
        np_image = sitk.GetArrayFromImage(sitk_image)
        # threshold image between p10 and p98 then re-scale [0-255]
        p0 = np_image.min().astype('float')
        p10 = np.percentile(np_image, 10)
        p99 = np.percentile(np_image, 99)
        p100 = np_image.max().astype('float')
        sitk_image = sitk.Threshold(sitk_image,
                                    lower=p10,
                                    upper=p100,
                                    outsideValue=p10)
        sitk_image = sitk.Threshold(sitk_image,
                                    lower=p0,
                                    upper=p99,
                                    outsideValue=p99)
        sitk_image = sitk.RescaleIntensity(sitk_image,
                                           outputMinimum=0,
                                           outputMaximum=255)
        from skimage import io

        # Convert from [depth, width, height] to [height, width, depth]
        image_data = sitk.GetArrayFromImage(sitk_image).transpose(2, 1, 0).astype('uint8')

        # rotate image
        image_rotated = img_rotation(np.expand_dims(image_data, axis=0))[0]
        rx, ry, rz = rotation_degrees
        tx, ty, tz = translations
        result_image = sitk.GetImageFromArray(image_rotated.transpose(2, 1, 0))
        result_image.CopyInformation(sitk_image)
        sitk.WriteImage(result_image, f'rx{rx}_ry{ry}_rz{rz}_tx{tx}_ty{ty}_tz{tz}.tif')
        # set the rotate center to the middle of the image
        image_dims = np.shape(image_data)
        center = [(image_dims[0] - 1) / 2, (image_dims[1] - 1) / 2 - 1, (image_dims[2] - 1) / 2]
        pos_rotation.SetCenter(center)

        # all_landmarks = getLandmarksFromTXTFile(landmark_path, split=",")
        all_landmarks = getLandmarksFromExcelFile(landmark_path, scale=0.5)
        all_landmarks_rotated = {}
        for lm_name in landmark_names:
            lm = all_landmarks[lm_name]
            lm_rotated = pos_rotation.TransformPoint(lm)

            # visualization
            sl = image_data[:, :, int(lm[2])].astype(np.uint8)
            sl_color = cv2.cvtColor(sl, cv2.COLOR_GRAY2RGB)
            cv2.circle(sl_color, (int(lm[1]), int(lm[0])), 2, (0, 255, 0))

            sl_rotated = image_rotated[:, :, int(lm_rotated[2])].astype(np.uint8)
            sl_rotated_color = cv2.cvtColor(sl_rotated, cv2.COLOR_GRAY2RGB)
            cv2.circle(sl_rotated_color, (int(lm_rotated[1]), int(lm_rotated[0])), 2, (0, 255, 0))

            cv2.imshow('image', sl_color)
            cv2.imshow('rotated', sl_rotated_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            all_landmarks_rotated.update({lm_name: lm_rotated})
        pd.DataFrame(all_landmarks_rotated).to_csv(f'rx{rx}_ry{ry}_rz{rz}_tx{tx}_ty{ty}_tz{tz}.csv')

