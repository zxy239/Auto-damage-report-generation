# Automated Tunnel Damage Report Generation

## Overview

Ageing infrastructure requires effective monitoring. Although reality capture technology allows for digital representation of as-is status of assets, the risk recognition and assessment process remains manual, subjective, and time-consuming. This paper focuses on Italian road tunnels and proposes a comprehensive, web-based automated framework that uses panoramic tunnel images as input to automatically generate damage records and risk assessment reports for tunnel assets. During the detection process, engineers prioritize ensuring that no damage is overlooked while placing less emphasis on the precise boundaries. 

To meet these requirements and address the complexities of real-world scenarios, we propose an Intersection over Union with buffer zone (IoUb) evaluation method. This approach aims to reduce inconsistencies in manual annotations and de-emphasize strict boundary precision, thereby enhancing the robustness and effectiveness of damage detection model evaluations. We evaluated several instance segmentation algorithms and ultimately recommend adopting a lower confidence threshold, as it better aligns with our application. We introduce post-processing methods that aggregate the predictions from multiple inferences to meet the demands of processing ultra-high-resolution (UHR) panoramic images. Finally, based on reliable detection results, we use a statistical ranking approach to categorize damage severity, generating the damage report in an end-to-end web-based platform. The proposed framework significantly enhances the efficiency of professionals in planning and monitoring ageing tunnel assets.

## Components

### IoU with buffer zone (IoUb)

We provided the code, which is based on the **cocoeval.py** module from the **pycocotools** library. The code has been modified to output instance segmentation evaluation based on **IoUb**, including category-specific results. It also prints evaluation results related to **AR (Average Recall)**.

### Web-based framework

We provide a **web-based fully automated framework** for processing high-resolution images, including **pre-processing and post-processing** for tunnel images requiring prediction, along with an **interactive interface** based on Gradio.

### Weight file

We provide the .pth file of Mask2Former that we used in our paper for predicting the tunnel defect.

## Usage

- The **IoUb-based evaluation** requires replacing the original **`cocoeval.py`** file. Users can set the **buffer range** in the code.

- We provide the entire framework in the form of a **Jupyter Notebook**, allowing users to conveniently extract the necessary code snippets. The part of the framework that calls damage prediction results uses **MMDetection**, but it can be replaced with any other inference framework. For the **Mask2Former** used in our framework, we made slight modifications to the **MMDetection** implementation to directly obtain the required logits. Specifically, at the end of `instance_postprocess` in `MaskFormerFusionHead`, we added the line `results.mask_logits = mask_pred` to directly obtain the logits. If using the **Transformers** framework, the required logits can be obtained directly.

## Progress

The complete code will be released alongside the preprint of the paper. All or part of the dataset may be made publicly available in the future as planned.

## Contact

For more information, please contact: zxy239@student.bham.ac.uk
