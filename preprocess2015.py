import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import glob
from batchgenerators.utilities.data_splitting import get_split_deterministic

from multiprocessing import Pool


try:
    import SimpleITK as sitk
except ImportError:
    print("You need to have SimpleITK installed to run this example!")
    raise ImportError("SimpleITK not found")


data_folder='../data/BRATS2015_Training'
preprocessed_data_folder='BRATS2015_precessed' 
train_ids_path='train_ids.pkl'
valid_ids_path='valid_ids.pkl'


def get_list_of_files(base_dir):
    """
    returns a list of lists containing the filenames. The outer list contains all training examples. Each entry in the
    outer list is again a list pointing to the files of that training example in the following order:
    T1, T1c, T2, FLAIR, segmentation
    :param base_dir:
    :return:
    """
    list_of_lists = []
    for glioma_type in ['HGG', 'LGG']:
        current_directory = join(base_dir, glioma_type)
        patients = subfolders(current_directory, join=False)
        for p in patients:
            patient_directory = join(current_directory, p)
            t1_file = join(patient_directory,'*_T1.*')
            t1_file=  join(t1_file,'*.mha')
            t2_file = join(patient_directory,'*_T2.*')
            t2_file=  join(t2_file,'*.mha')
            t1c_file = join(patient_directory,'*_T1c.*')
            t1c_file=  join(t1c_file,'*.mha')
            flair_file = join(patient_directory,'*_Flair.*')
            flair_file=  join(flair_file,'*.mha')
            seg_file = join(patient_directory,'*.OT.*')
            seg_file=  join(seg_file,'*.mha')
            case = [t1_file, t1c_file, t2_file, flair_file, seg_file]
            this_case=[]
            for i in case:
                try:
                    this_case.append(glob.glob(i)[0])
                except IndexError:
                    raise RuntimeError("Could not find file matching {}".format(i))
            list_of_lists.append(this_case)
    print("Found %d patients" % len(list_of_lists))
    return list_of_lists


def load_and_preprocess(case, patient_name, output_folder):
    """
    loads, preprocesses and saves a case
    This is what happens here:
    1) load all images and stack them to a 4d array
    2) crop to nonzero region, this removes unnecessary zero-valued regions and reduces computation time
    3) normalize the nonzero region with its mean and standard deviation
    4) save 4d tensor as numpy array. Also save metadata required to create niftis again (required for export
    of predictions)
    :param case:
    :param patient_name:
    :return:
    """
    # load SimpleITK Images
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get pixel arrays from SimpleITK images
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    # get some metadata
    spacing = imgs_sitk[0].GetSpacing()
    # the spacing returned by SimpleITK is in inverse order relative to the numpy array we receive. If we wanted to
    # resample the data and if the spacing was not isotropic (in BraTS all cases have already been resampled to 1x1x1mm
    # by the organizers) then we need to pay attention here. Therefore we bring the spacing into the correct order so
    # that spacing[0] actually corresponds to the spacing of the first axis of the numpy array
    spacing = np.array(spacing)[::-1]

    direction = imgs_sitk[0].GetDirection()
    origin = imgs_sitk[0].GetOrigin()

    original_shape = imgs_npy[0].shape

    # now stack the images into one 4d array, cast to float because we will get rounding problems if we don't
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)

    # now find the nonzero region and crop to that
    nonzero = [np.array(np.where(i != 0)) for i in imgs_npy]
    nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
    nonzero = np.array([np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]).T
    # nonzero now has shape 3, 2. It contains the (min, max) coordinate of nonzero voxels for each axis

    # now crop to nonzero
    imgs_npy = imgs_npy[:,
               nonzero[0, 0] : nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1,
               ]

    # now we create a brain mask that we use for normalization
    nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
    brain_mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    for i in range(len(nonzero_masks)):
        brain_mask = brain_mask | nonzero_masks[i]

    # now normalize each modality with its mean and standard deviation (computed within the brain mask)
    for i in range(len(imgs_npy) - 1):
        mean = imgs_npy[i][brain_mask].mean()
        std = imgs_npy[i][brain_mask].std()
        imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
        imgs_npy[i][brain_mask == 0] = 0

    print(imgs_npy.shape)
    # now save as npz
    np.save(join(output_folder, patient_name + ".npy"), imgs_npy)

    metadata = {
        'spacing': spacing,
        'direction': direction,
        'origin': origin,
        'original_shape': original_shape,
        'nonzero_region': nonzero,
        'name':patient_name
    }

    save_pickle(metadata, join(output_folder, patient_name + ".pkl"))



def save_segmentation_as_nifti(segmentation, metadata, output_file):
    original_shape = metadata['original_shape']
    seg_original_shape = np.zeros(original_shape, dtype=np.uint8)
    nonzero = metadata['nonzero_region']
    seg_original_shape[nonzero[0, 0] : nonzero[0, 1] + 1,
               nonzero[1, 0]: nonzero[1, 1] + 1,
               nonzero[2, 0]: nonzero[2, 1] + 1] = segmentation
    sitk_image = sitk.GetImageFromArray(seg_original_shape)
    sitk_image.SetDirection(metadata['direction'])
    sitk_image.SetOrigin(metadata['origin'])
    # remember to revert spacing back to sitk order again
    sitk_image.SetSpacing(tuple(metadata['spacing'][[2, 1, 0]]))
    sitk.WriteImage(sitk_image, output_file)


def get_list_of_patients(preprocessed_data_folder):
    npy_files = subfiles(preprocessed_data_folder, suffix=".npy", join=True)
    # remove npy file extension
    patients = [i[:-4] for i in npy_files]
    return patients





if __name__ == "__main__":
    list_of_lists = get_list_of_files(data_folder)
    maybe_mkdir_p(preprocessed_data_folder)

    patient_names=[]
    for case in list_of_lists:
        case=case[0].replace('\\','/')
        patient=case.split('/')[4]
        if case.split('/')[3]=='LGG':
            patient='LGG'+patient
        patient_names.append(patient)

    p = Pool(processes=8)
    p.starmap(load_and_preprocess, zip(list_of_lists, patient_names, [preprocessed_data_folder] * len(list_of_lists)))
    p.close()
    p.join()



    patients = get_list_of_patients(preprocessed_data_folder)
    train, val = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)
    train_file=open(join(preprocessed_data_folder,train_ids_path),'wb')
    valid_file=open(join(preprocessed_data_folder,valid_ids_path),'wb')

    for i,name in enumerate(train):
        train[i] = os.path.basename(name)
    for i,name in enumerate(val):
        val[i] = os.path.basename(name)

    pickle.dump(train,train_file)
    pickle.dump(val,valid_file)
    train_file.close()
    valid_file.close()

