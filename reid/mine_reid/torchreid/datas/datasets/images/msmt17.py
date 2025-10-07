import os
import glob

__all__ = [
    "MSMT17"
]


class MSMT17(object):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http: //www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = "MSMT17"

    def __init__(self, root="data", **kwargs):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')       # 32621
        self.query_dir = os.path.join(self.dataset_dir, 'query')                    # 11659
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')      # 82161

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> MSMT17 loaded")
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
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        datasets = list()
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pids = set()
        for img_path in img_paths:
            img_name = img_path.split(os.sep)[-1]
            infos = img_name.split('_')
            pid, camid = int(infos[0]), int(infos[1][1:])
            # ignore junk image
            if pid == -1:
                continue
            assert 0 <= pid <= 4101
            assert 1 <= camid <= 15
            pids.add(pid)
            camid -= 1  # camera id start from 0
            datasets.append([img_path, pid, camid])
        if relabel:
            pid2label = {pid: idx for idx, pid in enumerate(pids)}
            for p in datasets:
                p[1] = pid2label[p[1]]
        num_pids = len(pids)
        num_imgs = len(datasets)
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pids):
            assert idx == pid, "See code comment for explanation"
        return datasets, num_pids, num_imgs


if __name__ == "__main__":
    msmt = MSMT17(r"D:\workspace\data\dl\reid")

