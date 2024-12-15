import cv2
from matplotlib import pyplot as plt
from IPython.display import Image

# Load the images
funny_img = cv2.imread('images/Funny-picture.jpg', 1)
cb_img = cv2.imread('images/Funny-picture.jpg', cv2.IMREAD_GRAYSCALE)
cb_img_fuzzy = cv2.imread('images/Funny-picture.jpg', cv2.IMREAD_GRAYSCALE)

# Resize the image to a new resolution
new_resolution = (1900, 1080)
resized_img = cv2.resize(funny_img, new_resolution)

# Commented out the grayscale images display section
# Display the grayscale images using Matplotlib
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title('Grayscale Image')
# plt.imshow(cb_img, cmap='gray')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Grayscale Fuzzy Image')
# plt.imshow(cb_img_fuzzy, cmap='gray')
# plt.axis('off')

# plt.show()

# Split the image into the B, G, R components
img_NZ_bgr = cv2.imread("images/Funny-picture.jpg", cv2.IMREAD_COLOR)
if img_NZ_bgr is None:
    print("Error: Could not load image 'images/Funny-picture.jpg'")
else:
    b, g, r = cv2.split(img_NZ_bgr)

    # Show the RGB channels
    plt.figure(figsize=[20, 5])

    plt.subplot(1, 4, 1)
    plt.imshow(r, cmap="gray")
    plt.title("Red Channel")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(g, cmap="gray")
    plt.title("Green Channel")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(b, cmap="gray")
    plt.title("Blue Channel")
    plt.axis('off')

    # Merge the individual channels into a BGR image
    imgMerged = cv2.merge((b, g, r))

    # Show the merged output
    plt.subplot(1, 4, 4)
    plt.imshow(imgMerged[:, :, ::-1])
    plt.title("Merged Output")
    plt.axis('off')

    plt.show()

    # Convert the image to HSV
    img_hsv = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2HSV)

    # Split the image into the H, S, V components
    h, s, v = cv2.split(img_hsv)

    # Convert the original BGR image to RGB for display
    img_NZ_rgb = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2RGB)

    # Show the HSV channels
    plt.figure(figsize=[20, 5])

    plt.subplot(1, 4, 1)
    plt.imshow(h, cmap="gray")
    plt.title("H Channel")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(s, cmap="gray")
    plt.title("S Channel")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(v, cmap="gray")
    plt.title("V Channel")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(img_NZ_rgb)
    plt.title("Original")
    plt.axis('off')

    plt.show()

    # Save the image
    cv2.imwrite("funny_image_SAVED.png", img_NZ_bgr)

    # Display the saved image
    display(Image(filename='New_Zealand_Lake_SAVED.png'))

# Wait for a key press and close the OpenCV window
cv2.waitKey(0)
cv2.destroyAllWindows()