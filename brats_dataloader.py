from batchgenerators.augmentations.crop_and_pad_augmentations import crop
import numpy as np
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image,resize_image_by_padding
from batchgenerators.utilities.file_and_folder_operations import *

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image,resize_image_by_padding
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms import Compose
import torch
import math


class brats_dataloader(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.patch_size = patch_size
        self.num_modalities = 4
        self.indices = list(range(len(data)))


    @staticmethod
    def load_patient(patient):
        data = np.load(patient + ".npy", mmap_mode="r")
        metadata = load_pickle(patient + ".pkl")
        return data, metadata

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        patch_size=None
        if self.patch_size == None :
            shapes=[]
            for i in patients_for_batch:
                patient_data, _ = self.load_patient(i)
                shapes.append(patient_data[0].shape)
            patch_size = np.max(shapes,0)
            patch_size =(patch_size[0]+16-patch_size[0]%16,patch_size[1]+16-patch_size[1]%16,patch_size[2]+16-patch_size[2]%16)
        else:
            patch_size=self.patch_size

        if len(patients_for_batch)==self.batch_size:
            # initialize empty array for data and seg
            data = np.zeros((self.batch_size, self.num_modalities,*patch_size), dtype=np.float32)
            seg = np.zeros((self.batch_size, 1, *patch_size), dtype=np.float32)
        else:
            data = np.zeros((len(patients_for_batch), self.num_modalities,*patch_size), dtype=np.float32)
            seg = np.zeros((len(patients_for_batch), 1, *patch_size), dtype=np.float32)

        metadata = []
        patient_names = []

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            patient_data, patient_metadata = self.load_patient(j)
            if self.patch_size is not None:
                # this will only pad patient_data if its shape is smaller than self.patch_size
                patient_data = pad_nd_image(patient_data, patch_size)

                # now random crop to self.patch_size
                # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
                # dummy dimension in order for it to work (@Todo, could be improved)
                patient_data, patient_seg = crop(patient_data[:-1][None], patient_data[-1:][None], patch_size, crop_type="random")

                data[i] = patient_data[0]
                seg[i] = patient_seg[0]

            else:
                for k in range(len(patient_data)):
                    if k != len(patient_data)-1:
                        data[i][k] = resize_image_by_padding(patient_data[k],patch_size)
                    else:
                        seg[i][0] = resize_image_by_padding(patient_data[k],patch_size)

            metadata.append(patient_metadata)
            patient_names.append(j)
    

        return {'data': data, 'seg':seg, 'metadata':metadata, 'names':patient_names}

    def __len__(self):
        return int(math.ceil(len(self.indices)/self.batch_size))




def get_train_transform(patch_size):
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27%
    # of samples will be augmented, the rest will just be cropped
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # Gaussian Noise
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_file_list(folder,file_name):
    in_file=open(join(folder,file_name),'rb')
    data=pickle.load(in_file)
    in_file.close()
    for i ,name in enumerate(data):
        data[i]=join(folder,name)
    return data



def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[-3:])
    if  data.device.type=='cuda':
        return torch.zeros(new_shape).cuda().scatter_(1,data,1)
    else:
        return torch.zeros(new_shape).scatter_(1,data,1)
    

