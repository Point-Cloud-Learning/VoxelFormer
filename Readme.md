## VoxelFormer: An Efficient and Lightweight Point Cloud Analysis with Point-Voxel Attention

### Abstract

----------------------------
> The attention mechanism at the heart of Transformer is a set operator that is naturally applicable to point sets, and thus it is gradually emerging in point cloud analysis, achieving excellent results on a wide range of point cloud tasks. However, mainstream attention-based point cloud models suffer from high time and computational overheads as they employ the attention mechanism within the local neighborhood of each point, whereas point clouds often contain a large number of points. To this end, we apply the attention mechanism to voxels to explore an efficient and lightweight model VoxelFormer. Specifically, the smaller voxel feature set that inscribes the shape of a point cloud is first learned through the voxelization and the voxel encoder constructed from the scalar attention. Since the voxel encoder directly affects the quality of the voxel feature set, we enhance its effectiveness in terms of enriching input semantics and capturing position information by improving the MLP-based point embedding layer and designing the context relative position encoding, respectively. Then, the point-voxel attention layer is proposed to compute the offset attention between the point set and the voxel feature set to capture global shape features with an approximate linear complexity relative to the number of points. In addition, local features are important for point cloud analysis, and we therefore add the locality compensation module at the end to enable points to be perceptive of their local environments. Extensive experiments on point cloud segmentation and recognition demonstrate that VoxelFormer achieves comparable accuracies with efficient inference speed and less computational and memory resources compared to mainstream attention models.

### Pipeline 

---------------------------------------------

<img src="fig/Pipeline.png">


### Our novelties come from three aspects:

--------------------------------------------

> - An efficient and lightweight model VoxelFormer is designed by applying the attention mechanism to voxels, which achieves competitive accuracy with efficient inference and less computational and storage resources by transforming the attention computation from local neighborhoods over all points to finite voxel spaces.
> - The voxel encoder is constructed based on the scalar attention for extracting local geometric information in voxel spaces, and its effectiveness is enhanced in terms of enriching input semantics and capturing position information by improving the MLP-based point embedding layer and designing the context relative position encoding, respectively. 
> - The point voxel attention layer is proposed, which achieves capturing global information with an approximate linear complexity relative to the number of points through the attention interaction between the point set and the voxel feature set with a smaller base.


### Results

----------------------------

|    ShapeNet    |        | S3DIS  |        |      ModleNet40 |        |
|:---------------:|:------:|:--------------:|:------:|:------:|:------:|
|   Ins. mIoU    |  86.7  | mIoU   |  69.5  |      OA(%)      |  93.5  |
|   Cat. mIoU     |  83.9  |  OA  |  89.2  |    mAcc(%)     |  91.0  |
|   Inferences(ms) | 21.670 |  Inferences(ms)  | 81.230 |Inference(ms)  | 24.660 |
|       Params(M)    | 0.254  |  Params(M)  | 0.316  |Params(M)    | 0.252  |
|      FLOPs(G)    | 1.177  | FLOPs(G)  | 5.691  |  FLOPs(G)     | 1.166  |


