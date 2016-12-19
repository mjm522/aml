IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50
IMAGE_CHANNELS = 3
NUM_FP = 4

network_params = {
    'num_filters': [5, 5, NUM_FP],
    'batch_size': 25,
    'image_width': IMAGE_WIDTH,
    'image_height': IMAGE_HEIGHT,
    'image_channels': IMAGE_CHANNELS,
    'image_size': IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_CHANNELS
}