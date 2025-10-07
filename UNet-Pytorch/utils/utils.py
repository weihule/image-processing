import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes+1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i+1].set_title(f"Mask (class {i+1})")
        ax[i+1].imshow(mask == i)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def test():
    import cv2
    img_path = r"D:\workspace\data\carvana-image-masking-challenge\train\0cdf5b5d0ce1_01.jpg"
    mask_path = r"D:\workspace\data\carvana-image-masking-challenge\train_masks\0cdf5b5d0ce1_01_mask.gif"

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)  

    print(img.shape, mask.shape)
    
    plot_img_and_mask(img, mask)


if __name__ == "__main__":
    test()





