import os
import glob


__all__ = [
    'DukeMTMC'
]


class DukeMTMC:
    """
    Dataset statistics:
    identities: 1401 (train + query)
    images: 16522 (train) + 2228 (query) + 17661 (gallery)
    cameras: 8
    """
    dataset_dir = 'DukeMTMC-reID'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> DukeMTMC-reID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError(f'{self.dataset_dir} is not available')
        if not os.path.exists(self.train_dir):
            raise RuntimeError(f'{self.train_dir} is not available')
        if not os.path.exists(self.query_dir):
            raise RuntimeError(f'{self.query_dir} is not available')
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError(f'{self.gallery_dir} is not available')

    @staticmethod
    def _process_dir(dir_path, relabel=False):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pids = set()

        dataset = []
        for img_path in img_paths:
            img_name = img_path.split(os.sep)[-1]
            infos = img_name.split('_')
            pid, camid = int(infos[0]), int(infos[1][1])
            assert 1 <= camid <= 8
            pids.add(pid)
            camid -= 1
            dataset.append([img_path, pid, camid])

        if relabel:
            pid2label = {v: k for k, v in enumerate(pids)}
            for p in dataset:
                p[1] = pid2label[p[1]]

        num_pids = len(pids)
        num_images = len(dataset)

        return dataset, num_pids, num_images


if __name__ == "__main__":
    duke = DukeMTMC(root='D:\\workspace\\data\\dl\\reid')




