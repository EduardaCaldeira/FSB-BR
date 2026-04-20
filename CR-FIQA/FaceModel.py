import cv2
import numpy as np
from sklearn.preprocessing import normalize
import os
from skimage import transform as trans

class FaceModel():
    def __init__(self, model_prefix, model_epoch, ctx_id=7, backbone="iresnet100"):
        self.gpu_id=ctx_id
        self.image_size = (112, 112)
        self.model_prefix=model_prefix
        self.model_epoch=model_epoch
        self.model=self._get_model(ctx=ctx_id, image_size=self.image_size, prefix=self.model_prefix, epoch=self.model_epoch, layer='fc1', backbone=backbone)
    def _get_model(self, ctx, image_size, prefix, epoch, layer):
        pass

    def _getFeatureBlob(self, input_blob):
        pass

    def get_feature(self, image_path, color="BGR"):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (112, 112))
        if (color == "RGB"):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        a = np.transpose(image, (2, 0, 1))
        input_blob = np.expand_dims(a, axis=0)
        emb=self._getFeatureBlob(input_blob)
        emb = normalize(emb.reshape(1, -1))
        return emb

    def get_batch_feature(self, image_path_list, batch_size=16, color="BGR"):
        count = 0
        num_batch =  int(len(image_path_list) / batch_size)
        features = []
        quality_score=[]
        for i in range(0, len(image_path_list), batch_size):
            if count < num_batch:
                tmp_list = image_path_list[i : i+batch_size]
            else:
                tmp_list = image_path_list[i :]
            count += 1

            images = []
            for image_path in tmp_list:
                image = cv2.imread(image_path)
                image = cv2.resize(image, (112, 112))
                if (color=="RGB"):
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                a = np.transpose(image, (2, 0, 1))
                images.append(a)
            input_blob = np.array(images)

            emb, qs = self._getFeatureBlob(input_blob)
            quality_score.append(qs)
            features.append(emb)
            #print("batch"+str(i))
        features = np.vstack(features)
        quality_score=np.vstack(quality_score)
        features = normalize(features)
        return features, quality_score

    def get_aligned_ijbc_feature(self, img_dir, landmark_file, batch_size=16, color="BGR"):
        features = []
        quality_score=[]

        landmark_list = open(landmark_file)
        landmark_lines = landmark_list.readlines()

        for start_num in range(0, len(landmark_lines), batch_size):
            batch_ldmk = landmark_lines[start_num:start_num+batch_size]
            images = []
            for _, each_line in enumerate(batch_ldmk):
                name_lmk_score = each_line.strip().split(' ')
                img_name = os.path.join(img_dir, name_lmk_score[0])
                image = cv2.imread(img_name)

                lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
                lmk = lmk.reshape((5, 2))

                image = self.align_ijbc(image, lmk)
                image = cv2.resize(image, (112, 112))
                if (color=="RGB"):
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                a = np.transpose(image, (2, 0, 1))
                images.append(a)

            input_blob = np.array(images)
            emb, qs = self._getFeatureBlob(input_blob)
            quality_score.append(qs)
            features.append(emb)

        features = np.vstack(features)
        quality_score = np.vstack(quality_score)
        features = normalize(features)

        return features, quality_score

    def align_ijbc(self, rimg, landmark):
        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2

        src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0

        image_size = self.image_size
        
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                                M, (image_size[1], image_size[0]),
                                borderValue=0.0)
        return img