# Spatiotemporal Contrastive Learning for Echocardiography View Classification

This repository accompanies the published conference paper:

**Spatiotemporal Contrastive Learning for Echocardiography View Classification**  
Preshen Naidoo, Patricia Fernandes, Isreal Ufumaka, Nasim Dadashi Serej, James Howard, Darrel P. Francis, Charlotte H. Manisty, Massoud Zolgharni  
In *Artificial Intelligence in Healthcare (AIiH 2025)*,  
Lecture Notes in Computer Science (LNCS), Springer, 2025  
https://doi.org/10.1007/978-3-032-00656-1_18

The complete source code will be released alongside the final archival version of the paper.

---

## Overview

Accurate echocardiographic view classification is essential for reliable downstream cardiac analysis, yet remains challenging due to anatomical overlap, operator variability, temporal phase differences, and class imbalance.

This work proposes a **spatiotemporal contrastive learning framework** that integrates temporal and spatial augmentation strategies to learn robust and invariant feature representations from echocardiography videos. The proposed method achieves **expert-level performance (96.4% accuracy)** and demonstrates improved robustness compared to existing supervised and self-supervised approaches.

---

## Key Contributions

- Introduces a **temporal rolling augmentation strategy** to promote cardiac phase invariance  
- Proposes a **spatiotemporal contrastive learning framework** for echocardiography video classification  
- Demonstrates improved feature separability using supervised contrastive learning  
- Achieves performance comparable to, and in some cases exceeding, inter-observer agreement between clinical experts  

---

## Requirements

The codebase was developed and tested with the following dependencies:

- **Python** 3.10  
- **TensorFlow** 2.15.0  
- **Keras** 2.15.0  
- **Pandas** 1.5.3  
- **NumPy** 1.24.4  
- **OpenCV** 4.5.5.64  

> Using the specified versions is recommended for reproducibility.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{naidoo2025spatiotemporal,
  title={Spatiotemporal Contrastive Learning for Echocardiography View Classification},
  author={Naidoo, Preshen and Fernandes, Patricia and Ufumaka, Isreal and Dadashi Serej, Nasim and Howard, James and Francis, Darrel P. and Manisty, Charlotte and Zolgharni, Massoud},
  booktitle={Artificial Intelligence in Healthcare},
  series={Lecture Notes in Computer Science},
  year={2025},
  publisher={Springer},
  doi={10.1007/978-3-032-00656-1_18}
}
