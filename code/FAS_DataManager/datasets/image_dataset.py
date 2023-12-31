import os
import cv2
import torch
from torchvision import transforms

def crop_face_with_mtcnn_bbox_info(image, mtcnn_bbox_file_path):
    # You can define your own processing function
    with open(mtcnn_bbox_file_path, 'r') as f:
        bbox_info = f.read()
        bboxes, landmark_points = bbox_info.split()
        bboxes = bboxes.split(',')
        landmark_points = landmark_points.split(',')
        x1, y1, x2, y2 = int(bboxes[0]), int(bboxes[1]), int(bboxes[2]), int(bboxes[3])
        if x1<0:
            x1=1
        if y1<0:
            y1=1

    face = image[y1:y2, x1:x2]
    return face

class WFDD:
    def __init__(self, image_folder_dir, spoofing_label, transform, num_frames=1000):

        os.listdir(image_folder_dir)
        image_list = os.listdir(image_folder_dir)
        self.image_list = [os.path.join(image_folder_dir, p) for p in image_list]
        self.spoofing_label = spoofing_label


    def __getitem__(self, index):
        im = self.__read_image__(index) # cv2 image, format [H, W, C], BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im = im.transpose((2,0,1))
        tensor = self.transform(im)
        tensor = tensor.to(torch.float)
        target = {
            'spoofing_label':  self.spoofing_label
        }
        return index, tensor, target, self.zip_file_path



class ImageDataset:

    def __init__(self, file_list, label_list, torchvision_transform=None, use_original_frame=False, bbox_suffix='_bbox_mtccnn.txt'):
        self.image_list = file_list
        self.label_list = label_list
        self.use_original_frame = use_original_frame
        self.bbox_suffix = bbox_suffix

        if torchvision_transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = torchvision_transform

        self.len = len(self.image_list)


    def load_image(self, image_path):

        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.use_original_frame:
            return im
        else:
            # Crop face
            if '.png' in image_path:
                bbox_file_path = image_path.replace('.png', self.bbox_suffix)
            elif '.jpg' in image_path:
                bbox_file_path = image_path.replace('.jpg', self.bbox_suffix)
            elif '.tif' in image_path:
                bbox_file_path = image_path.replace('.tif', self.bbox_suffix)
            else:
                bbox_file_path = image_path[0:-4] + self.bbox_suffix

            if os.path.exists(bbox_file_path):
                try:
                    crop_face = crop_face_with_mtcnn_bbox_info(im, bbox_file_path)
                    # cv2.imwrite('test.png', crop_face)
                    return crop_face
                except:
                    return im
            else:
                return im

    def __getitem__(self, index):
        path = self.image_list[index]
        spoofing_label = int(self.label_list[index])
        img = self.load_image(path) # cv2 image: numpy, (H,W,C), uint8[0,255],
        img_tensor = self.transform(img) # tensor: (C,H,W), float [0-1]
        img_tensor = img_tensor.to(torch.float32)

        target = {
            'spoofing_label': spoofing_label
        }

        if '.mp4' in path.lower() or '.avi' in path.lower() or '.mov' in path.lower():
            video_id = path.split('/')
            video_id = os.path.join(*video_id)

        else:
            video_id = path


        return index, img_tensor, target, video_id, cv2.resize(img, (224,224))

    def __len__(self):
        return self.len

if __name__ == '__main__':

    dummy_image_list = ['/home/rizhao/data/FAS/frames/CASIA-FASD/train_release/9/9.png']
    dummy_label_list = [1]
    dataset = ImageDataset(dummy_image_list, dummy_label_list, bbox_suffix='_bbox_mtccnn.txt')
    out = dataset.__getitem__(0)
    import IPython; IPython.embed()