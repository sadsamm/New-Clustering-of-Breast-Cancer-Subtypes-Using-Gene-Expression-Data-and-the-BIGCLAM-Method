# New-Clustering-of-Breast-Cancer-Subtypes-Using-Gene-Expression-Data-and-the-BIGCLAM-Method
"Code for clustering breast cancer subtypes via BIGCLAM using gene expression data. Includes preprocessing, network construction, &amp; community detection (Python). Reproducible pipelines, visualizations, and analysis for precision oncology research."

## Project Status  
✅ **Completed Research Project**  

## Features  
- **Dataset**: Implemented on the [GSE96058](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058) breast cancer gene expression dataset.  
- **GPU-Accelerated BIGCLAM**: Optimized implementation of the BIGCLAM algorithm for GPU computation (PyTorch/CUDA).  
- **Preprocessing Pipeline**: Automated normalization and feature selection for RNA-seq data.  
- **Enhanced Clustering**: Modified affinity propagation for improved subtype identification.  
 
---

## Getting Started  

### Requirements  
- Python 3.10+  
- PyTorch (with CUDA for GPU support)  
- numpy, pandas, json, pickle, networkx, seaborn, scikit-learn, matplotlib  

### Usage  
1. Download GSE96058 data from GEO and place in `/content/drive/MyDrive/gene_expression_data_target_added.csv`.  
2. Run the code

## Project Inspiration
This project builds on the methodology from the original BIGCLAM paper:
Community Affiliation Model for Big Networks (Leskovec et al., 2013).

## Contributing
Contributions are welcome! If you have suggestions or find issues, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
Special thanks to my supervisors Dr. Mohammad Teshnehlab and Dr. Saeed Pirmoradi for their guidance and support.
