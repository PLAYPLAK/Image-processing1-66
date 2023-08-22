import cv2
import numpy as np
import matplotlib.pyplot as plt
################################################################################
base_img = cv2.imread("lab2/chainat.jpg")

h_o_r_1 = cv2.calcHist([base_img], [0], None, [256], [0, 256])/(base_img.shape[0]*base_img.shape[1])
h_o_g_1 = cv2.calcHist([base_img], [1], None, [256], [0, 256])/(base_img.shape[0]*base_img.shape[1])
h_o_b_1 = cv2.calcHist([base_img], [2], None, [256], [0, 256])/(base_img.shape[0]*base_img.shape[1])

cdf_r = h_o_r_1.copy()
cdf_g = h_o_g_1.copy()
cdf_b = h_o_b_1.copy()

for i in range(1,256):
    cdf_r[i]   = cdf_r[i] + cdf_r[i-1]
    cdf_g[i] = cdf_g[i] + cdf_g[i-1]
    cdf_b[i]  = cdf_b[i] + cdf_b[i-1]

################################################################################
template_img = cv2.imread("lab2/kmitl.jpg")

h_o_r_2 = cv2.calcHist([template_img], [0], None, [256], [0, 256])/(template_img.shape[0]*template_img.shape[1])
h_o_g_2 = cv2.calcHist([template_img], [1], None, [256], [0, 256])/(template_img.shape[0]*template_img.shape[1])
h_o_b_2 = cv2.calcHist([template_img], [2], None, [256], [0, 256])/(template_img.shape[0]*template_img.shape[1])

cdf_r_temp   = h_o_r_2.copy()
cdf_g_temp = h_o_g_2.copy()
cdf_b_temp  = h_o_b_2.copy()

for i in range(1,256):
    cdf_r_temp[i]   = cdf_r_temp[i] + cdf_r_temp[i-1]
    cdf_g_temp[i] = cdf_g_temp[i] + cdf_g_temp[i-1]
    cdf_b_temp[i]  = cdf_b_temp[i] + cdf_b_temp[i-1]

################################################################################


new_img_r = np.zeros(256)
new_img_g = np.zeros(256)
new_img_b = np.zeros(256)

for i in range(0,256):
    diff = abs(cdf_r[i]-cdf_r_temp)
    new_img_r[i] = np.argmin(diff)
    diff = abs(cdf_g[i]-cdf_g_temp)
    new_img_g[i] = np.argmin(diff)
    diff = abs(cdf_b[i]-cdf_b_temp)
    new_img_b[i] = np.argmin(diff)

new_img = np.zeros(base_img.shape)
for i in range(0,base_img.shape[0]):
    for j in range(0,base_img.shape[1]):
        new_img[i,j,0]=new_img_r[base_img[i,j,0]]
        new_img[i,j,1]=new_img_g[base_img[i,j,1]]
        new_img[i,j,2]=new_img_b[base_img[i,j,2]]

new_img= new_img.astype(np.uint8)
new_img_h_r = cv2.calcHist([new_img], [0], None, [256], [0, 256])/(new_img.shape[0]*new_img.shape[1])
new_img_h_g = cv2.calcHist([new_img], [1], None, [256], [0, 256])/(new_img.shape[0]*new_img.shape[1])
new_img_h_b = cv2.calcHist([new_img], [2], None, [256], [0, 256])/(new_img.shape[0]*new_img.shape[1])

cdf_r_new = new_img_h_r.copy()
cdf_g_new = new_img_h_g.copy()
cdf_b_new = new_img_h_b.copy()

for i in range(1,256):
    cdf_r_new[i] = cdf_r_new[i]+cdf_r_new[i-1]
    cdf_g_new[i] = cdf_g_new[i]+cdf_g_new[i-1]    
    cdf_b_new[i] = cdf_b_new[i]+cdf_b_new[i-1]



#############################################################################
plt.figure(figsize=(12, 10))

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
plt.title('Original image')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.plot(h_o_r_1, color='red', label='Red')
plt.plot(h_o_g_1, color='green', label='Green')
plt.plot(h_o_b_1, color='blue', label='Blue')
plt.legend()

plt.subplot(3, 3, 3)
plt.plot(cdf_r, color='red', label='Red')
plt.plot(cdf_g, color='green', label='Green')
plt.plot(cdf_b, color='blue', label='Blue')
plt.legend()

plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(template_img, cv2.COLOR_BGR2RGB))
plt.title('Templare image')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.plot(h_o_r_2, color='red', label='Red')
plt.plot(h_o_g_2, color='green', label='Green')
plt.plot(h_o_b_2, color='blue', label='Blue')
plt.legend()

plt.subplot(3, 3, 6)
plt.plot(cdf_r_temp, color='red', label='Red')
plt.plot(cdf_g_temp, color='green', label='Green')
plt.plot(cdf_b_temp, color='blue', label='Blue')
plt.legend()

plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
plt.title('Image Match')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.plot(new_img_h_r, color='red', label='Red')
plt.plot(new_img_h_g, color='green', label='Green')
plt.plot(new_img_h_b, color='blue', label='Blue')
plt.legend()

plt.subplot(3, 3, 9)
plt.plot(cdf_r_new, color='red', label='Red')
plt.plot(cdf_g_new, color='green', label='Green')
plt.plot(cdf_b_new, color='blue', label='Blue')
plt.legend()

plt.tight_layout()
plt.show()