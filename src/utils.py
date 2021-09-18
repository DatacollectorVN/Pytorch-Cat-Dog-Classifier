import numpy as np 

def resize_image(image, size):
    ''' Resize image with size (rectangle or square)
    args:
        image: (PIL.Image object)
        size: (tuple) (new_width, new_heigh)
    '''

    [w, h] = image.size
    [new_w, new_h] = size[0], size[1]
    resized_image = image.resize((new_w, new_h))
    
    return resized_image

def crop_square(image):
    ''' Square crop image
    args:
        image: (PIL.Image object)
    '''

    [w, h] = image.size

    if w == h:
        return image
    elif w < h:
        return image.resize((w, w))
    else:
        return image.resize((h, h))


class EarlyStopping():
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def onehot_encoder(label, num_objects):
    ''' Encode label into onehot label
    Args:
        label: (list) the object of image (class_ids)
        num_object: (int) the total number of objects in the dataset
    '''

    onehot_label = np.zeros(num_objects).astype(np.int32)
    if not label[0] == 0:
        onehot_label[np.array(label)-1] = 1
    
    return onehot_label.tolist()

