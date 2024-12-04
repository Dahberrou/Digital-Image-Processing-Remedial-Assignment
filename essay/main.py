import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to detect and highlight red objects in an image
def detect_red_color(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the range for detecting the color red in HSV space
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    
    # Create a mask to detect red color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Define the second range for detecting red color (in case of red being in the other half of the HSV space)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    # Combine the two masks to capture the entire red range
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply the mask to the original image to highlight the red areas
    red_detected = cv2.bitwise_and(image, image, mask=red_mask)
    
    # Count the number of red pixels (this will give us an idea of the area covered by red)
    red_area = np.sum(red_mask) / 255  # Divide by 255 to get the count of white pixels (red pixels in mask)
    
    # Display the results
    plt.figure(figsize=(10, 8))
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Show the red-detected image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(red_detected, cv2.COLOR_BGR2RGB))
    plt.title('Red Color Detection')
    plt.axis('off')
    
    # Print the area of the detected red region (in terms of number of red pixels)
    print(f"Total Red Area (in pixels): {red_area}")
    
    # Show the plots
    plt.tight_layout()
    plt.show()

# Run the function on the provided image
image_path = 'photo.png'  # Replace this with the path to your image file
detect_red_color(image_path)
