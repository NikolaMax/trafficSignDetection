import cv2

PATH_TO_DATA = "data"
LABELS_FILE_NAME = "labels.csv"
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
BATCH_SIZE = 50
EPOCHS_NUMBER = 1
STEPS_PER_EPOCH = 2000

# Model constants
NO_OF_FILTERS = 60
IMAGE_DIMENSIONS = (32, 32, 3)
NO_OF_NODES = 500 # broj cvorova u Hidden (Dense) sloju
SIZE_OF_POOL = (2, 2) # povecavanje generalizacije, redukovanje moguceg overfitting-a
SIZE_OF_FILTER = (5, 5)
SIZE_OF_FILTER2 = (3, 3) # uklanja 2 pixela sa svake strane kad se koriste 32, 32 slike

# Camera constants
FONT = cv2.FONT_HERSHEY_DUPLEX
BRIGHTNESS = 180
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
THRESHOLD = 0.75
