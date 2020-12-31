# BU-Net: Brain Tumor Segmentation Using Modified U-Net Architecture

This repo implements the Brain Tumor Segementation Paper published in MDPI journal you can access it [here](https://www.mdpi.com/2079-9292/9/12/2203).

## Abstract

The semantic segmentation of a brain tumor is of paramount importance for its treatment and prevention. Recently, researches have proposed various neural network-based architectures to improve the performance of segmentation of brain tumor sub-regions. Brain tumor segmentation, being a challenging area of research, requires improvement in its performance. This paper proposes a 2D image segmentation method, BU-Net, to contribute to brain tumor segmentation research. Residual extended skip (RES) and wide context (WC) are used along with the customized loss function in the baseline U-Net architecture. The modifications contribute by finding more diverse features, by increasing the valid receptive field. The contextual information is extracted with the aggregating features to get better segmentation performance. The proposed BU-Net was evaluated on the high-grade glioma (HGG) datasets of the BraTS2017 Challenge—the test datasets of the BraTS 2017 and 2018 Challenge datasets. Three major labels to segmented were tumor core (TC), whole tumor (WT), and enhancing core (EC). To compare the performance quantitatively, the dice score was utilized. The proposed BU-Net outperformed the existing state-of-the-art techniques. The high performing BU-Net can have a great contribution to researchers from the field of bioinformatics and medicine.
(same as paper)*

## Dataset
You can get detailed information regarding dataset from theri official [page](https://www.med.upenn.edu/cbica/brats2020/data.html)
Figure below shows the two sample cases from the dataset. The BraTS 2018 dataset contains similar training images as that of BraTS 2017. The labeling procedure and classes remain the same. The only difference being made in BraTS 2018 is of the validation dataset. A new validation dataset is made available which carried images collected from 66 patients from unknown grade.
![alt text](https://github.com/Mr-TalhaIlyas/Brain-Tumor-Segmentation/blob/master/screens/img1.png)

## Netwrok Architecture
Overall architecture of the proposed BU-Net including RES blocks and wide context block.
![alt text](https://github.com/Mr-TalhaIlyas/Brain-Tumor-Segmentation/blob/master/screens/img2.png)
The architecture for Residual Extended Skip (RES) block.
![alt text](https://github.com/Mr-TalhaIlyas/Brain-Tumor-Segmentation/blob/master/screens/img3.png)
The architecture for the Wide Context (WC) block.
![alt text](https://github.com/Mr-TalhaIlyas/Brain-Tumor-Segmentation/blob/master/screens/img4.png)

## Visual Results
The qualitative analysis of BU-Net and U-Net. There are 3 colors used to represent three different tumor classes. Red represents necrosis and non-enhancing; green represents edema; and yellow represents an enhancing tumor.
![alt text](img5.png)

