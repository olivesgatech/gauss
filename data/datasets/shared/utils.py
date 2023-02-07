class DatasetStructure:
    def __init__(self):
        # different sets
        self.train_set = None
        self.train_labels = None
        self.test_set = None
        self.test_labels = None

        # set statistics
        self.train_len = None
        self.test_len = None
        self.num_classes = None
        self.img_size = None
        self.is_configured = False

