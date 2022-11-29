# A Transformer-based Decoder for Semantic Segmentation with Multi-level Context Mining

Official implementation of the paper [A Transformer-based Decoder for Semantic Segmentation with Multi-level Context Mining](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/383_ECCV_2022_paper.php),  
by Bowen Shi*, Dongsheng Jiang*, Xiaopeng Zhang, Han Li, Wenrui Dai, Junni Zou, Hongkai Xiong, Qi Tian. 

[[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880617.pdf)] [[Appendix](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880617-supp.pdf)]

<div  align="center">
<img src="./imgs/framework.png" alt="framework" align=center />
</div>

## Installation

Our code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/). For install and data preparation, please refer to the guidelines in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/).

Our code is still in preparation. Currently it can be used by moving the provided code files to the corresponding directory of MMSegmentation.

## Training 
Example: train SegFormer-B1 + SegDeformer on ADE20K:

python start_local_train.py --config_file segformer/segformer_mit-b1_512x512_160k_ade20k_segdeformer3.py

## Citation
If you find this repository/work helpful in your research, welcome to cite the paper.
```
@inproceedings{segdeformer,
    title={A Transformer-based Decoder for Semantic Segmentation with Multi-level Context Mining}, 
    author={Bowen Shi and Dongsheng Jiang and Xiaopeng Zhang and Han Li and Wenrui Dai and Junni Zou and Hongkai Xiong and Qi Tian},
    journal={European Conference on Computer Vision},
    year={2022}
}
