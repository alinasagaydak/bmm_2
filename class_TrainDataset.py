from torch.utils.data import DataLoader, Dataset

class TrainDataset (Dataset):
    def __init__(self, annotation_file):
        super().__init__()
        self.train_df = annotation_file
        self.image_ids = self.train_df["id"].unique()
    
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        bboxes = self.train_df[self.train_df["id"]==image_id]
        df[df.id==index]
        img = cv2.imread("C:\\Users\\alina\\jupiter\\Wider_face_dataset_2\\wider_face_train\\" + str(index) +".jpg")
        
        return img


test = TrainDataset(df)
test.__getitem__(0).shape
