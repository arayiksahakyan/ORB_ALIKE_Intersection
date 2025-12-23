import matplotlib.pyplot as plt
import numpy as np

experiments = ['ORB', 'ORB + ALIKE']

keypoints = [5000, 1446]
inliers = [2435, 870]
reproj_error = [1.2276, 1.1997]

x = np.arange(len(experiments))
width = 0.35

fig, ax1 = plt.subplots()

ax1.bar(x - width/2, keypoints, width, label='Keypoints')
ax1.bar(x + width/2, inliers, width, label='Inliers')
ax1.set_ylabel('Count')
ax1.set_xticks(x)
ax1.set_xticklabels(experiments)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(x, reproj_error, marker='o', linewidth=2, label='Reprojection Error')
ax2.set_ylabel('Reprojection Error (px)')
ax2.legend(loc='upper right')

plt.title('Comparison of ORB vs ORB + ALIKE')
plt.tight_layout()
plt.show()

