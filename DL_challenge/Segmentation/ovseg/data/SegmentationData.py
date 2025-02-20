from KCD.Segmentation.ovseg.data.DataBase import DataBase
from KCD.Segmentation.ovseg.data.SegmentationDataloader import SegmentationDataloader, SegmentationDataloader_regress
from os import listdir
from os.path import join


class SegmentationData(DataBase):

    def __init__(self, augmentation=None, use_double_bias=False, *args, **kwargs):
        self.augmentation = augmentation
        self.use_double_bias = use_double_bias
        super().__init__(*args, **kwargs)

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            
            if self.use_double_bias:
                raise NotImplementedError
            else:
                self.trn_dl = SegmentationDataloader(self.trn_ds,
                                                     augmentation=self.augmentation,
                                                     **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            try:
                if self.use_double_bias:
                    raise NotImplementedError
                else:
                    self.val_dl = SegmentationDataloader(self.val_ds,
                                                         augmentation=self.augmentation,
                                                         **self.val_dl_params)
                    
            except (AttributeError, TypeError):
                print('No validatation dataloader initialised')
                self.val_dl = None

    def clean(self):
        self.trn_dl.dataset._maybe_clean_stored_data()
        if self.val_dl is not None:
            self.val_dl.dataset._maybe_clean_stored_data()


class SegmentationData_trainontest(DataBase):

    def __init__(self, augmentation=None, use_double_bias=False, *args, **kwargs):
        self.augmentation = augmentation
        self.use_double_bias = use_double_bias
        super().__init__(*args, **kwargs)

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            train_path_dict = self.trn_ds.path_dicts
            test_path_dict = self.val_ds.path_dicts
            self.trn_ds.path_dicts = train_path_dict + test_path_dict

            if self.use_double_bias:
                raise NotImplementedError
            else:
                self.trn_dl = SegmentationDataloader(self.trn_ds,
                                                     augmentation=self.augmentation,
                                                     **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            try:
                if self.use_double_bias:
                    raise NotImplementedError
                else:
                    self.val_dl = SegmentationDataloader(self.val_ds,
                                                         augmentation=self.augmentation,
                                                         **self.val_dl_params)

            except (AttributeError, TypeError):
                print('No validatation dataloader initialised')
                self.val_dl = None

    def clean(self):
        self.trn_dl.dataset._maybe_clean_stored_data()
        if self.val_dl is not None:
            self.val_dl.dataset._maybe_clean_stored_data()

class SegmentationData_regress(DataBase):

    def __init__(self, augmentation=None, use_double_bias=False,
                 regress_key = 'noises',*args, **kwargs):
        self.augmentation = augmentation
        self.use_double_bias = use_double_bias
        self.reg_key = regress_key
        super().__init__(*args, **kwargs)

    def initialise_dataloader(self, is_train):
        if is_train:
            print('Initialise training dataloader')
            train_path_dict = self.trn_ds.path_dicts
            test_path_dict = self.val_ds.path_dicts
            self.trn_ds.path_dicts = train_path_dict + test_path_dict

            if self.use_double_bias:
                raise NotImplementedError
            else:
                self.trn_dl = SegmentationDataloader_regress(self.trn_ds,
                                                     augmentation=self.augmentation,
                                                     regress_key = self.reg_key,
                                                     **self.trn_dl_params)
        else:
            print('Initialise validation dataloader')
            try:
                if self.use_double_bias:
                    raise NotImplementedError
                else:
                    self.val_dl = SegmentationDataloader(self.val_ds,
                                                         augmentation=self.augmentation,
                                                         **self.val_dl_params)

            except (AttributeError, TypeError):
                print('No validatation dataloader initialised')
                self.val_dl = None

    def clean(self):
        self.trn_dl.dataset._maybe_clean_stored_data()
        if self.val_dl is not None:
            self.val_dl.dataset._maybe_clean_stored_data()
