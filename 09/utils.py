import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

def show_image(img_paths, save_name, rows=4, cols=3): 
    assert len(img_paths) <= rows*cols
    plt.figure(figsize=(15, 15))
    grid = gridspec.GridSpec(rows, cols) 

    for idx, img_path in enumerate(img_paths):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(grid[idx])
        ax.imshow(image)
    plt.savefig(save_name)

def show_pie(length, labels):
    mpl.rc('font', size=13)
    plt.figure(figsize=(10, 10))
    plt.pie(length, labels=labels, autopct='%.1f%%')
    plt.savefig('pie.jpg')