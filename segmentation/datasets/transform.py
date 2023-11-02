import numpy as np


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, annots, scale, size = sample['image'], sample['annots'], sample[
            'scale'], sample['size']

        if annots.shape[0] == 0:
            return sample

        if np.random.uniform(0, 1) < self.prob:
            image = image[:, ::-1, :]
            _, w, _ = image.shape
            # Ë®Æ½·­×ªÍ¼Ïñ
            image = np.flip(image, axis=1)
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0] = w - x2
            annots[:, 2] = w - x1

        return {
            'image': image,
            'annots': annots,
            'scale': scale,
            'size': size,
        }