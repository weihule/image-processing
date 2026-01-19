import os
import os.path
from pathlib import Path
import hashlib
import errno
from tqdm import tqdm

def gen_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not Path(fpath).is_file():
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), 'b'):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def makedir_exist_ok(dirpath):
    Path(dirpath).mkdir(exist_ok=True, parents=True)


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str): Name to save the file under. If None, use the basename of the URL
        md5 (str): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
                )

def list_dir(root, prefix=False):
    dirs = [p for p in Path(root).iterdir() if p.is_dir()]
    if prefix is True:
        return [p.as_posix() for p in dirs]
    else:
        return [p.name for p in dirs]


def list_files(root, suffix, prefix=False):
    """
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
    """
    if isinstance(suffix, str):
        pattern = f'*{suffix}'
    elif isinstance(suffix, tuple):
        # 元组后缀，转换成 glob 支持的 {a,b} 格式
        suffix_clean = [s.lstrip('.') for s in suffix]
        pattern = f'*.{{{",".join(suffix_clean)}}}'  # 最终生成 *.{jpg,png}
    else:
        raise TypeError(f'suffix 必须是 str 或 tuple 类型，当前是 {type(suffix)}')
    files = Path(root).glob(pattern)
    if prefix:
        # 返回完整路径的字符串
        file_list = [file.as_posix() for file in files]
    else:
        # 仅返回文件名
        file_list = [file.name for file in files]

    return file_list


def test():
    root = r'D:\workspace\data\images\test_images'
    dirs = list_files(root, suffix='.jpg', prefix=False)
    print(dirs, len(dirs))


if __name__ == "__main__":
    test()




