
PATH_TO_DATA = "data"
LABELS_FILE_NAME = "labels.csv"
BATCH_SIZE = 50
STEPS_PER_EPOCH = 2000
EPOCHS_NUMBER = 1
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2

# model constants
IMAGE_DIMENSIONS = (32, 32, 3)
NO_OF_FILTERS = 60
SIZE_OF_FILTER = (5, 5)
SIZE_OF_FILTER2 = (3, 3) # uklanja 2 pixela sa svake strane kad se koriste 32, 32 slike
SIZE_OF_POOL = (2, 2) # povecavanje generalizacije, redukovanje moguceg overfitting-a
NO_OF_NODES = 500 # broj cvorova u Hidden (Dense) sloju