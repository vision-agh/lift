
# LiFT: <ins>**Li**</ins>ghtweight, <ins>**F**</ins>PGA-<ins>**T**</ins>ailored 3D object detection based on LiDAR data

This is the official implementation of ***LiFT*** (DASIP 2025). LiFT is a simple, lightweight (20.73 GMAC) and fully-sparse 3D object detector.
It is tailored for (but not restricted to) real-time implementation on low-end FPGAs.
By the time of publication, it achieves the best detection performance on NuScenes-val among methods with comparable structure and complexity.
For more details, please refer to:

**LiFT: Lightweight, FPGA-tailored 3D object detection based on LiDAR data [[Paper](https://arxiv.org/abs/2501.11159)]** <br />
[Konrad Lis](https://orcid.org/0000-0003-2034-0590), [Tomasz Kryjak](https://orcid.org/0000-0001-6798-4444), [Marek Gorgoń](https://orcid.org/0000-0003-1746-1279)<br />

## Installation

The 
Refer to the original [OpenPCDet readme](README_openpcdet.md). 
The additional requirement in LiFT is the *Brevitas*, which can be installed with `pip install brevitas`.


## Usage

The **LiFT** is trained and validated in the same way as all the other models in the OpenPCDet repo.
Refer to the original [OpenPCDet readme](README_openpcdet.md).

## Experimental results

| nuScenes Detection      |  Set |  mAP |  NDS |   Download  |
|---------------|:----:|:----:|:----:|:-----------:|
| [LiFT](tools/cfgs/nuscenes_models/lift_int8.yaml)     |  val | 51.84 | 61.01 | [ckpt](https://drive.google.com/file/d/1CsiVwpYfyonLufTWx5riH48lshwSuMv-/view?usp=sharing) |


## Remarks

* The implementation on FPGA is not ready yet, but it's in a progress. Updates regarding its release will be posted in this repository.
* The LiFT is quantised using the [Brevitas](https://github.com/Xilinx/brevitas) library, as we regard it's more flexible than native PyTorch and spconv quantisation and provides more compatible interface to [FINN](https://github.com/Xilinx/finn) for future implementation on FPGA.


## Acknowledgement
-  This work is built upon the [VoxelNeXt](https://github.com/dvlab-research/VoxelNeXt), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [spconv](https://github.com/traveller59/spconv). 


## License

This project is released under the [Apache 2.0 license](LICENSE).