import os
import glob


class Market1501:
    """
    dataset statistics:
    identities: 1501 (1 for background)
    images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501/Market-1501-v15.09.15'

    def __init__(self, root, **kwargs):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
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
        """
        check if all files are available before going deeper
        Returns:
        """
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError('{} is not available'.format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError('{} is not available'.format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError('{} is not available'.format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError('{} is not available'.format(self.gallery_dir))

    @staticmethod
    def _process_dir(dir_path, relabel=False):
        datasets = list()
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pids = set()
        for img_path in img_paths:
            img_name = img_path.split(os.sep)[-1]
            infos = img_name.split('_')
            pid, camid = int(infos[0]), int(infos[1][1])
            # ignore junk image
            if pid == -1:
                continue
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            pids.add(pid)
            camid -= 1  # camera id start from 0
            datasets.append([img_path, pid, camid])
        if relabel:
            pid2label = {pid: idx for idx, pid in enumerate(pids)}
            for p in datasets:
                p[1] = pid2label[p[1]]
        num_pids = len(pids)
        num_imgs = len(datasets)

        return datasets, num_pids, num_imgs


if __name__ == "__main__":
    mark = Market1501(root='D:\\workspace\\data\\dl')




