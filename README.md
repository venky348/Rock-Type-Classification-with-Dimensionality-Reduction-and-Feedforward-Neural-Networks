# Rock Clustering and Classification Using Machine Learning Techniques

This project implements a pipeline for classifying and visualizing various types of rocks using machine learning techniques. It integrates a Feedforward Neural Network (FNN) for classification after dimensionality reduction using PCA, t-SNE, LLE, and MDS.

## Features
- **Dimensionality Reduction**: Applies PCA, t-SNE, LLE, and MDS to preprocess and reduce the dimensionality of rock image data.
- **Image Processing**: Processes rock images by resizing, normalizing, and flattening to make them suitable for machine learning models.
- **Neural Network Classification**: Uses a Feedforward Neural Network to classify rocks based on their reduced feature sets.
- **Visualization**: Generates 2D scatter plots to demonstrate clusters of rock types in reduced dimensions.

## Methods Used

### 1. Dimensionality Reduction
Dimensionality reduction techniques preprocess the image data to reduce computational complexity and extract meaningful patterns:
- **PCA**: Captures global variance and reduces features to 116 components while retaining 90% variance.
- **t-SNE**: Emphasizes local structures for better cluster separations.
- **LLE**: Preserves local relationships within data points.
- **MDS**: Represents pairwise distances between data points.

### 2. Feedforward Neural Network (FNN)
The Feedforward Neural Network is used for classification:
- **Architecture**: Includes multiple dense layers with ReLU activations for feature learning and classification.
- **Input Data**: Takes reduced feature sets obtained from dimensionality reduction methods as input.
- **Output**: Predicts the class (Igneous, Metamorphic, Sedimentary) of each rock image.
- **Optimization**: Uses the Adam optimizer for fast convergence, with categorical cross-entropy as the loss function.
- **Training**: Data is split into training and testing sets to evaluate the model's performance.

### 3. Model Performance
- **Accuracy**: The FNN achieves high accuracy when trained on PCA-reduced features, as they capture global patterns efficiently.
- **Comparison**: Features from t-SNE and LLE also yield good results, emphasizing the strength of these techniques for clustering and classification.
- **Challenges**: Overlap in feature space between some rock categories may slightly affect classification performance.

## Observations and Insights
- **Dimensionality Reduction Impact**: PCA-reduced features work best with the FNN due to their compact and global representation.
- **Rock Category Clustering**:
  - **Igneous Rocks**: Form distinct clusters, making them easier to classify.
  - **Metamorphic Rocks**: Overlap slightly with sedimentary rocks due to transitional textures.
  - **Sedimentary Rocks**: Display variability but are generally well-clustered.
- **Neural Network Efficiency**: The FNN demonstrates robust performance when paired with effective dimensionality reduction techniques.

