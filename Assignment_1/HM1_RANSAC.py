import numpy as np
from utils import draw_save_plane_with_points, normalize


if __name__ == "__main__":


    np.random.seed(0)
    # load data, total 130 points inlcuding 100 inliers and 30 outliers
    # to simplify this problem, we provide the number of inliers and outliers here

    noise_points = np.loadtxt("HM1_ransac_points.txt")


    #RANSAC
    # Please formulate the palnace function as:  A*x+B*y+C*z+D=0     

    w = 1 - 30 / noise_points.shape[0] # fraction of inliers
    n = 3 # the minimal number of points needed to define a plane
    p = 0.999 # the probability of at least one hypothesis does not contain any outliers
    k = int(np.ceil(np.log(1-p) / np.log(1-w**n))) # samples chosen high enough to keep 1-p below a desired failure rate
    sample_time = k # the minimal time that can guarantee the probability of at least one hypothesis does not contain any outliers is larger than 99.9%
    distance_threshold = 0.05

    # sample points group
    rng = np.random.default_rng()
    sample_idx = rng.choice(noise_points.shape[0], size=(k, n), replace=False)
    sampled_group = noise_points[sample_idx].reshape(k, n, 3)

    # estimate the plane with sampled points group
    # use cross product to calculate the normal of the plane
    v1 = sampled_group[:, 1, :] - sampled_group[:, 0, :]
    v2 = sampled_group[:, 2, :] - sampled_group[:, 0, :]
    normals = np.cross(v1, v2) # (A, B, C)
    Ds = -np.sum(normals * sampled_group[:, 0, :], axis=1) # D = -Ax - By - Cz
    A_B_C_Ds = np.concatenate([normals, Ds[:, None]], axis=1)

    #evaluate inliers (with point-to-plance distance < distance_threshold)
    distances = np.abs(noise_points @ normals.T + Ds) / np.linalg.norm(normals) # (130, k)
    inlier_idx = distances < distance_threshold
    inlier_num = np.sum(inlier_idx, axis=0) #inliers num for each hypothesis
    # print(inlier_num)
    # select the hypothesis with the largest number of inliers as your best hypothesis
    best_hypo_idx = np.argmax(inlier_num)
    best_hypo_inliers_idx = inlier_idx[:, best_hypo_idx]
    best_hypo_inliers_num = inlier_num[best_hypo_idx]
    best_hypo_inliers = noise_points[best_hypo_inliers_idx]

    # minimize the sum of squared perpendicular distances of all inliers with least-squared method 
    A  = np.concatenate([best_hypo_inliers, np.ones((best_hypo_inliers_num, 1))], axis=1)
    U, D, Vt = np.linalg.svd(A)
    pf = Vt[-1, :] # (A, B, C, D)


    # draw the estimated plane with points and save the results 
    # check the utils.py for more details
    # pf: [A,B,C,D] contains the parameters of palnace function  A*x+B*y+C*z+D=0  
    pf = normalize(pf)
    draw_save_plane_with_points(pf, noise_points,"result/HM1_RANSAC_fig.png") 
    np.savetxt("result/HM1_RANSAC_plane.txt", pf)
    np.savetxt('result/HM1_RANSAC_sample_time.txt', np.array([sample_time]))
