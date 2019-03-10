class Config:
    exemplar_size = 127                    # exemplar size
    instance_size = 255                    # instance size
    score_size = 17
    eps = 0.01
    lr = 1e-2
    epoch = 50
    loss_huber_delta = 1.0
    loss_lamda = 1.0
    train_image_path = '/media/seok/My Passport/yt_bb/data/train'
    val_image_path = '/media/seok/My Passport/yt_bb/data/valid'
    csv_path = '/media/seok/My Passport/yt_bb/csv/all.csv'
    batch_size = 32
    frame_select_range = 100


config = Config()
