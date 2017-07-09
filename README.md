### Create virtual environment and install all dependencies 
### Assume conda environment
```
conda create -n testyolo python=3.5 -y
source activate testyolo
conda clean --tarballs --packages -y
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp35-cp35m-linux_x86_64.whl
conda install -c menpo opencv3=3.1.0 -y
conda install cython -y
```
### Clone github and install application 
```
git clone https://github.com/johnsonice/yolomodel.git
```

### Build application 
```
cd yolomodel 
python setup.py build_ext --inplace
```
### Download weights 
```
mkdir bin
cd bin 
wget https://www.dropbox.com/s/qhdbu7w3bi0tmyt/yolo.weights
wget https://www.dropbox.com/s/5xmhad7j7tqnt1m/tiny-yolo.weights
cd ..
```
### run demo 
#### if you can access to a camera 
```
./flow --model cfg/yolo.cfg --load bin/yolo.weights --demo camera --gpu .9 --threshold 0.4
```
#### work on images 
```
./flow --imgdir sample_img/ --model cfg/yolo.cfg --load bin/yolo.weights --gpu .9
```

### Access yolo model in python script
```python
from yolomodel import yolo_model
# create yolo instance 
yolo=yolo_model.yolo()
# preload weights
yolo.load(model='tiny-yolo',threshold=0.22)  ## now you can choose either load 'tiny-yolo' or 'yolo' model, and you can also pass in threshold 
# run demo if you have a camera
yolo.demo('camera')  # use key [ESC] to exit demo
# predict a image
result = yolo.predict('dog.jpg')
```

#### process images in batch or single image in memory return list of json
```python
import cv2

img_path = 'sample_img'
imgs = os.listdir(img_path)
cv_imgs = [cv2.imread(os.path.join(img_path,f),cv2.IMREAD_COLOR) for f in imgs] 
yolo=yolo_model.yolo();
yolo.load(model='tiny-yolo',threshold=0.22)  ## now you can choose either load 'tiny-yolo' or 'yolo' model, and you can also pass in threshold

##### predict single memory image
yolo.predict_imgcv(cv_imgs[0])

##### predict images in batches
inputs = cv_imgs
outputs = yolo.predict_imgcv_list(inputs,threshold=0.3)  ## now user can pass in threshold, if not, defaults to 0.35
print(outputs)
```
