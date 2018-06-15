import numpy as np

class MaskGenerator(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def gen(self, n):
        self.masks = np.ones((n, self.height, self.width))
        return self.masks

class CenterMaskGenerator(MaskGenerator):

    def __init__(self, height, width, ratio=0.5):
        super().__init__(height, width)
        self.ratio = ratio

    def gen(self, n):
        self.masks = np.ones((n, self.height, self.width))
        c_height = int(self.height * self.ratio)
        c_width = int(self.width * self.ratio)
        height_offset = (self.height - c_height) // 2
        width_offset = (self.width - c_width) // 2
        self.masks[:, height_offset:height_offset+c_height, width_offset:width_offset+c_width] = 0
        return self.masks

class RectangleMaskGenerator(MaskGenerator):

    def __init__(self, height, width,  rec=None):
        super().__init__(height, width)
        if rec is None:
            rec = int(0.25*self.height), int(0.75*self.width), int(0.75*self.height), int(0.25*self.width)
        self.rec = rec

    def gen(self, n):
        top, right, bottom, left = self.rec
        self.masks = np.ones((n, self.height, self.width))
        self.masks[:, top:bottom, left:right] = 0
        return self.masks

class MultiRectangleMaskGenerator(MaskGenerator):

    def __init__(self, height, width,  recs=None):
        super().__init__(height, width)
        if recs is None:
            rec = int(0.25*self.height), int(0.75*self.width), int(0.75*self.height), int(0.25*self.width)
            recs = [rec]
        self.recs = recs

    def gen(self, n):
        self.masks = np.ones((n, self.height, self.width))
        for rec in self.recs:
            top, right, bottom, left = rec
            self.masks[:, top:bottom, left:right] = 0
        return self.masks


class RandomRectangleMaskGenerator(MaskGenerator):

    def __init__(self, height, width, min_ratio=0.25, max_ratio=0.75, margin_ratio=0., batch_same=False):
        super().__init__(height, width)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.margin_ratio = margin_ratio
        self.batch_same = batch_same

    def gen(self, n):
        self.masks = np.ones((n, self.height, self.width))
        for i in range(self.masks.shape[0]):
            min_height = int(self.height * self.min_ratio)
            min_width = int(self.width * self.min_ratio)
            max_height = int(self.height * self.max_ratio)
            max_width = int(self.width * self.max_ratio)
            margin_height = int(self.height * self.margin_ratio)
            margin_width = int(self.width * self.margin_ratio)
            rng = np.random.RandomState(None)
            c_height = rng.randint(low=min_height, high=max_height)
            c_width = rng.randint(low=min_width, high=max_width)
            height_offset = rng.randint(low=margin_height, high=self.height-margin_height-c_height)
            width_offset = rng.randint(low=margin_width, high=self.width-margin_width-c_width)
            self.masks[i, height_offset:height_offset+c_height, width_offset:width_offset+c_width] = 0
        if self.batch_same:
            self.masks = np.stack([self.masks[i].copy() for i in range(n)], axis=0)
        return self.masks





def get_generator(name, size):
    if name=='full':
        return CenterMaskGenerator(size, size, ratio=1.0)
    elif name=='transparent':
        return CenterMaskGenerator(size, size, ratio=0.0)
    elif name=='center':
        return CenterMaskGenerator(size, size, ratio=0.5)
    elif name=='eye':
        return RectangleMaskGenerator(size, size, rec=[9, 27, 21, 5])
    elif name=='mouth':
        return RectangleMaskGenerator(size, size, rec=[22, 28, 32, 4])
        # return RectangleMaskGenerator(size, size, rec=[22, 32, 32, 0])
    elif name=='nose':
        return RectangleMaskGenerator(size, size, rec=[18, 21, 27, 11])
    elif name=='hair':
        return RectangleMaskGenerator(size, size, rec=[0,32,12,0])
        # return MultiRectangleMaskGenerator(size, size, recs=[[0,32,10,0], [0,9,32,0], [0,23,32,32]])
    elif name=='face':
        return RectangleMaskGenerator(size, size, rec=[8, 24, 32, 8])
    elif name=='bottom half':
        return RectangleMaskGenerator(size, size, rec=[16, 32, 32, 0])
    elif name=='bottom quarter':
        return RectangleMaskGenerator(size, size, rec=[24, 32, 32, 0])
    elif name=='random rec':
        return RandomRectangleMaskGenerator(size, size, min_ratio=1./8, max_ratio=(1.-1./8))



# class RandomShapeMaskGenerator(MaskGenerator):
#
#     def __init__(self, height, width):
#         super().__init__(height, width)
#
#     def _gen_rec_range(self, n, min_ratio=0.25, max_ratio=0.75, margin_ratio=0.):
#         recs = np.ones((n, self.height, self.width))
#         for i in range(recs.shape[0]):
#             min_height = int(self.height * min_ratio)
#             min_width = int(self.width * min_ratio)
#             max_height = int(self.height * max_ratio)
#             max_width = int(self.width * max_ratio)
#             margin_height = int(self.height * margin_ratio)
#             margin_width = int(self.width * margin_ratio)
#             rng = np.random.RandomState(None)
#             c_height = rng.randint(low=min_height, high=max_height)
#             c_width = rng.randint(low=min_width, high=max_width)
#             height_offset = rng.randint(low=margin_height, high=self.height-margin_height-c_height)
#             width_offset = rng.randint(low=margin_width, high=self.width-margin_width-c_width)
#             recs[i, height_offset:height_offset+c_height, width_offset:width_offset+c_width] = 0
#         recs =
#
#
#     def gen(self, n):
#         pass
