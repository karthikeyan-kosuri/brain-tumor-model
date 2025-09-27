# Dataset paths
train_images= 'data/segmentation_task/train/images'
train_mask='data/segmentation_task/train/masks'
test_images='data/segmentation_task/test/images'
test_mask='data/segmentation_task/test/images'

#Training parameters
epochs=50
batch_size=2
learning_rate=0.0001
num_worker=4

#Model parameters
in_channel=1
out_channel=1
init_features=64