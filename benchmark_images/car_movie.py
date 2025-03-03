import os
import cv2

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def read_images_from_folder(folder_path):
    images = []
    for i in range(len(os.listdir(folder_path))):
        filename = f"car{i:03d}.png"
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            img = cv2.imread(file_path)
            if img is not None:
                images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return images

def create_movie(images, output_path, fps=5):
    fig = plt.figure()
    ims = []

    for img in images:
        im = plt.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True)
    ani.save(output_path, writer='ffmpeg')

if __name__ == "__main__":
    folder_path = 'benchmark_images'
    output_path = 'benchmark_movie.mp4'
    images = read_images_from_folder(folder_path)
    create_movie(images, output_path)