import torch


class Options():
    """
    Derived from the base Config class and overrides specific values
    """

    name = "Options"
    phases = ['train']
    root_dir_test = '/media/tariqbdair/Semi-supervised/MALC_Testing/'
    root_dir = '/media/tariqbdair/Semi-supervised'
    log_path = '/media/tariqbdair/logs/'
    checkpoint_dir = "/media/tariqbdair/models"  # dir to save checkpoints
    checkpoint_dir_train = '/home/tariq/code/UsedData/models/lb100'
    ExpName = 'mixMatch'

    label_data_folder = "L3"
    unlabel_data_folder = "U9"
    validation_data_folder = "V3"

    img_width = 256
    img_height = 256
    img_channel = 1
    num_workers = 4     	# number of threads for data loading
    shuffle = True      	# shuffle the data set
    batch_size = 8     	# GTX1060 3G Memory
    mini_batch_size = 2
    learning_rate = 0.002  # learning rage
    weight_decay = 1e-4  # weight decay

    n_gpu = torch.cuda.device_count()  # number of GPUs
    pin_memory = True  # use pinned (page-locked) memory. when using CUDA, set to True
    is_cuda = torch.cuda.is_available()  # True --> GPU
    dtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type

    n_classes = 28
    n_filters = 64
    alpha = 0.75
    epochs = 100
    checkpoint_iter = 500
    beta = 1
    T = 0.5
    ema_decay = 0.999
    lambda_u = 75
    lambda_u2 = 75
    is_colab = False
    rampup_length = 4200
    max_lr = 0.1
    initial_lr = 0
    lr_rampup = 0
    lr_rampdown_epochs = 40
    consistency = 100
    consistency_rampupv= 30

    labels = ["Background", "Left Cortical WM", "Left Cortical GM", "Right Cortical WM", "Right Cortical GM",
              "Left Lateral Ventricle", "Left Cerebellar WM", "Left Cerebellar GM", "Left Thalamus", "Left Caudate",
              "Left Putamen", "Left Pallidum", "3rd Ventricle", "4th Ventricle", "Brain Stem", "Left Hippocampus",
              "Left Amygdala", "Left Ventral DC", "Right Lateral Ventricle", "Right Cerebellar WM",
              "Right Cerebellar GM", "Right Thalamus", "Right Caudate", "Right Putamen", "Right Pallidum",
              "Right Hippocampus", "Right Amygdala", "Right Ventral DC"]

