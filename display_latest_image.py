import time
import os
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "images/latest_image.png"


def load_and_display_image():
    """Continuously loads and displays the latest image if it changes."""
    last_modified = None
    plt.ion() 
    fig, ax = plt.subplots()
    img_display = None

    while True:
        if os.path.exists(IMAGE_PATH):
            modified_time = os.path.getmtime(IMAGE_PATH)  # Get last modified timestamp

            if last_modified is None or modified_time > last_modified:
                last_modified = modified_time
                try:
                    image = plt.imread(IMAGE_PATH)
                except:
                    print("Error loading image, try again")
                    continue
                if img_display is None:
                    img_display = ax.imshow(image, cmap="gray")
                    plt.axis("off")
                else:
                    img_display.set_data(image)
                    print("Updated image")
                plt.draw()
                plt.pause(0.1)  # Allow the display to update

        time.sleep(0.5)  # Check for updates every 0.5 seconds


if __name__ == "__main__":
    load_and_display_image()
