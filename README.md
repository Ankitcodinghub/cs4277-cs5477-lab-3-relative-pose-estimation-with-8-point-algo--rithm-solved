# cs4277-cs5477-lab-3-relative-pose-estimation-with-8-point-algo--rithm-solved
**TO GET THIS SOLUTION VISIT:** [CS4277/CS5477 Lab 3-Relative pose estimation with 8-point algo- rithm Solved](https://www.ankitcodinghub.com/product/cs4277-cs5477-lab-3-relative-pose-estimation-with-8-point-algo-rithm-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;94848&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS4277\/CS5477 Lab 3-Relative pose estimation with 8-point algo- rithm Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
1 CS4277/CS5477 Lab 3-1: Relative pose estimation with 8-point algo- rithm

<pre> In [2]: import cv2
         import matplotlib.pyplot as plt
</pre>
<pre>         import numpy as np
         import h5py
         import scipy.io as sio
         from eight_point import compute_fundamental, compute_essential, decompose_e
         %matplotlib inline
</pre>
1.0.1 Introduction

In this assignment, you will get to estimate the essential and fundamental matrix by using eight point algorithm. As discussed in the lecture, images taken from different views should fulfill the epipolar constraint, which can be used to estimate the fundamental and essential matrix. You will first estimate the fundamental and essential matrix with 15 correspondences provided in the dataset. Then you will decompose the essential matrix to find rotation and translation between two views. The decomposition will give 4 feasible camera poses and you will select the the correct pose by chriality check.

This assignment is worth 10% of the final grade.

References: * Lecture 6

Optional references: * Richard I. Hartley. In Defence of the 8-point Algorithm

1.0.2 Instructions

This workbook provides the instructions for the assignment, and facilitates the running of your code and visualization of the results. For each part of the assignment, you are required to complete the implementations of certain functions in the accompanying python file (eight_point.py).

To facilitate implementation and grading, all your work is to be done in that file, and you only have to submit the .py file.

Please note the following:

1. Fill in your name, email, and NUSNET ID at the top of the python file. 2. The parts you need to implement are clearly marked with the following:

<pre>```
""" YOUR CODE STARTS HERE """
</pre>
<pre>""" YOUR CODE ENDS HERE """
```
</pre>
<pre>, and you should write your code in between the above two lines.
</pre>
3. Note that for each part, there may certain functions that are prohibited to be used. It is important NOT to use those prohibited functions (or other functions with similar func- tionality). If you are unsure whether a particular function is allowed, feel free to ask any of the TAs.

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="section">
<div class="layoutArea">
<div class="column">
1.0.3 Submission Instructions

Zip your completed pnp.py and eight_point.py and upload onto the relevant work bin in Lumi- nus.

1.1 Part 1: Load and Visualize Data

In this part, you will get yourself familiar with the data by visualizing it. The data includes two images of the same object (im1.jpg and img2.jpg) and 15 correpondences (correspondences.mat). You can visualize the data with the provided code below.

<pre> In [2]: correspondences = sio.loadmat('data/correspondences_ud')
         data1_ori = correspondences['movingPoints']
         data2_ori = correspondences['fixedPoints']
         data1 = np.concatenate([data1_ori.T, np.ones((1, data1_ori.shape[0]))], axis = 0)
         data2 = np.concatenate([data2_ori.T, np.ones((1, data2_ori.shape[0]))], axis = 0)
         img1 = plt.imread('data/img1.jpg')
</pre>
<pre>         img2 = plt.imread('data/img2.jpg')
         plt.figure(figsize=(12, 6))
         plt.subplot(1, 2, 1)
         for j in range(data1_ori.shape[0]):
</pre>
<pre>             cv2.circle(img1, (np.int32(data1_ori[j, 0]), np.int32(data1_ori[j, 1])) , 5, (255,
         plt.imshow(img1)
         plt.subplot(1, 2, 2)
         for j in range(data2_ori.shape[0]):
</pre>
<pre>             cv2.circle(img2, (np.int32(data2_ori[j, 0]), np.int32(data2_ori[j, 1])) , 5, (255,
         plt.imshow(img2)
</pre>
<pre>Out[2]: &lt;matplotlib.image.AxesImage at 0x7ffb8b257208&gt;
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
0

0

</div>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
1.2 Part 2: Estimate Fundamental Matrix from Point Correspondences

In this part, you will implement the 8-point algorithm to estimate the fundamental matrix. For any pair of matching points xi ↔ x′i in two images, the 3 × 3 fundamental matrix is defined by the equation:

x′TFx = 0

Let f be the 9-vector made up of the entries of F in row-major order, we get:

(x′x, x′y, x′, y′x, y′y, y′, x, y, 1)f = 0

From a set of n point matches, we obtain a set of linear equations of the form:

Af = 0

The solution for f is the singlar vector corresponding to the smallest singular value of A. Then you will enforce the singularity constraint to F matrix such that the rank of F is 2. Note that the normalization step is very important here to for accurate estimation.

You can verify your estimation by visualizing the epipolar lines in both images, where the epiploar lines will pass through all matching points. The helper function plot_epipolar_line() is provided for visualization

Implement the following function(s): cv2.findFundamentalMat() * Prohibited Functions: cv2.findFundamentalMat()

* You may use the following functions: np.linalg.svd()

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>In [3]: F = compute_fundamental(data1, data2)
        plt.figure(figsize = (12, 6))
</pre>
<pre>        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        for i in range(data1.shape[1]):
</pre>
<pre>          plt.plot(data1[0, i], data1[1, i], 'bo')
          m, n = img1.shape[:2]
          line1 = np.dot(F.T, data2[:, i])
          t = np.linspace(0, n, 100)
</pre>
<pre>          lt1 = np.array([(line1[2] + line1[0] * tt) / (-line1[1]) for tt in t])
          ndx = (lt1 &gt;= 0) &amp; (lt1 &lt; m)
          plt.plot(t[ndx], lt1[ndx], linewidth=2)
</pre>
<pre>        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        for i in range(data2.shape[1]):
</pre>
<pre>          plt.plot(data2[0, i], data2[1, i], 'ro')
          m, n = img2.shape[:2]
          line2 = np.dot(F, data1[:, i])
          t = np.linspace(0, n, 100)
</pre>
<pre>          lt2 = np.array([(line2[2] + line2[0] * tt) / (-line2[1]) for tt in t])
          ndx = (lt2 &gt;= 0) &amp; (lt2 &lt; m)
          plt.plot(t[ndx], lt2[ndx], linewidth=2)
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
1.3 Part 3: Estimate Essential Matrix from Point Correspondences

In this part, you will also implement the 8-point algorithm to estimate the essential matrix. The steps are the same with the fundamental matrix estimation except for that :

1. The normalization step: For each correspondence xi ↔ x′i, compute K−1xi ↔ K′−1x′i. K and K′ are the camera calibration matrices which are given in the intrinsics.h5 file. Note we only give one camera calibration matrix here because the two images are taken by the same camera.

2. The singlarity constraint: The essential matrix should have two similar singular values, and third is zero.

Implement the following function(s): cv2.findEssentialMat()

* Prohibited Functions: cv2.findEssentialMat()

* You may use the following functions: np.linalg.svd(), np.linalg.inv()

Note that the your estimated essential matrix may be different from the results estimated by

using cv2.findEssentialMat(), because the cv2.findEssentialMat() use a different algorithm.

1.4 Part 4: Two-view Relative Pose Estimation

In this part, you will extract the relative rotaion R and translation t from the essential matrix E

accordint to:

E = ⌊t⌋×R.

The essentrial matrix can be decomposed into 4 feasible camera poses, and you will select the correct one by cheriality check. Specifically, the 3D structure can be computed with the linear tri- angulation method, and the 3D points should appear in front of both cameras. Note that we assum that the rotation and translation of the first camera are identity matrix and zeros respectively.

</div>
</div>
<div class="section">
<div class="layoutArea">
<div class="column">
<pre>In [4]: with h5py.File('data/intrinsics.h5', 'r') as f:
            K = f['K'][:]
</pre>
<pre>        E = compute_essential(data1, data2, K)
</pre>
</div>
</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
Implement the following function(s): cv2.recoverPose() * Prohibited Functions: cv2.recoverPose()

* You may use the following functions: np.linalg.svd()

<pre>In [5]: trans = decompose_e(E, K, data1, data2)
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
5

</div>
</div>
</div>
