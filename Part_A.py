# 1. Load an image using OpenCV and display it
import cv2 as cv
# reading the images
img =cv.imread(r"image\$50,000-Picsart-AiImageEnhancer.png") 
cv.imshow("Image",img)
cv.waitKey(0)

# 2. Convert the image to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # gray sclae 2 dimension black and white(1,0), 

#3. Display the image dimensions (height, width, channels)
print("Image Dimension",img.shape)
h,w,c = img.shape

# resized =cv.resize(img,(200,100))
print("Height:",h)
print("width:",w)
print("channel:",c)

# 4. Show the color histograms for RGB channels

if img is None:
    print("Image is not found")
    exit()
# convert from BGR to RGB 
img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
color = ('r', 'g', 'b')
for i, color in enumerate(color):
    hist = cv.calcHist([img_rgb],[i], None, [256],[0,256])
    plt.plot(hist,color=color)
    plt.xlim([0,256])

# Add lables and title
plt.title('Color HIstrogram for RGB Channel')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# 5. Convert the image to different color spaces:


# Convert from BGR (default in OpenCV) to RGB for display
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Convert to HSV color space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Convert to LAB color space
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

# Convert HSV and LAB to RGB for proper visualization in matplotlib
hsv_rgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
lab_rgb = cv.cvtColor(lab, cv.COLOR_LAB2RGB)

# Display all color spaces
titles = ['Original (RGB)', 'HSV', 'LAB']
images = [img_rgb, hsv_rgb, lab_rgb]

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.show()



# 









