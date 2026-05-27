import os 
import numpy as np
import cv2
from pycocotools.coco import COCO
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence

# --- CONFIG ---
DATA_DIR = r"C:\Users\monal\Desktop\coco\coco2017"
IMG_DIR = os.path.join(DATA_DIR, 'val2017')
ANN_FILE = os.path.join(DATA_DIR, 'annotations/instances_val2017.json')
MODEL_DIR = 'model'
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'segmentation_model.h5')
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 10
CLASSES = ['person', 'dog', 'cat']

class CocoKerasGenerator(Sequence):
    def __init__(self, img_dir, ann_file, batch_size, img_size, classes):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        
        # --- FIX: Load IDs for ANY category (Union), not ALL (Intersection) ---
        self.cat_ids = self.coco.getCatIds(catNms=classes)
        
        all_ids = []
        for cat_id in self.cat_ids:
            # Get images that have this specific category
            ids = self.coco.getImgIds(catIds=[cat_id])
            all_ids.extend(ids)
            
        # Remove duplicates (e.g., an image with a dog AND a person)
        self.img_ids = list(set(all_ids))
        
        print(f"Found {len(self.img_ids)} images containing {classes}")
        self.indexes = np.arange(len(self.img_ids))

    def __len__(self):
        # Calculate number of batches
        return int(np.floor(len(self.img_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_img_ids = [self.img_ids[k] for k in indexes]
        
        X = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32)

        for i, img_id in enumerate(batch_img_ids):
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            
            # Robust image loading
            img = cv2.imread(img_path)
            if img is None:
                continue # Skip missing images
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Load Mask
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)
            mask = np.zeros((img_info['height'], img_info['width']))
            for ann in anns:
                mask = np.maximum(mask, self.coco.annToMask(ann))
            
            # Resize
            X[i] = cv2.resize(img, (self.img_size, self.img_size)) / 255.0
            mask_resized = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            y[i] = mask_resized[:, :, np.newaxis]
            
        return X, y

def build_unet(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = layers.concatenate([u5, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def train():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    print("Loading Data Generator...")
    if not os.path.exists(IMG_DIR) or not os.path.exists(ANN_FILE):
        print(f"CRITICAL ERROR: Data not found.")
        print(f"Looking for images in: {IMG_DIR}")
        print(f"Looking for annotations in: {ANN_FILE}")
        return

    try:
        train_gen = CocoKerasGenerator(IMG_DIR, ANN_FILE, BATCH_SIZE, IMG_SIZE, CLASSES)
    except Exception as e:
        print(f"Error initializing generator: {e}")
        return
        
    if len(train_gen) == 0:
        print("ERROR: Dataset length is 0. This means no images matched the categories or Batch Size is too big.")
        return

    print("Building Model...")
    model = build_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Starting Training...")
    model.fit(train_gen, epochs=EPOCHS)
    
    model.save(MODEL_SAVE_PATH)
    print(f"SUCCESS: Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()