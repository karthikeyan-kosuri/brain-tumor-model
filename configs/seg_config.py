# Dataset paths
train_images= 'data/segmentation_task/train/images'
train_mask='data/segmentation_task/train/masks'
test_images='data/segmentation_task/test/images'
test_mask='data/segmentation_task/test/masks'

#Training parameters
epochs=50
batch_size=12
learning_rate=0.0001
num_worker=4

#Model parameters
in_channel=1
out_channel=1
init_features=64

#patience
patience=10
min_delta=1e-4