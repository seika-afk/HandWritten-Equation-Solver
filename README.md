# Equation Solver Data Preparation

This repository contains a Jupyter notebook focused on the initial data preparation and feature engineering steps for an image-based equation solver project. The primary goal of this notebook is to ingest a dataset of mathematical symbols (digits and operators), preprocess these images, and prepare them for subsequent machine learning model training.

**Note:** This project is currently a work in progress. This notebook covers the data ingestion and feature engineering phase, and further development will include model training and the full equation solving functionality.

## Table of Contents

- [Project Title](#project-title)
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Project Status](#project-status)

---

## Project Title

**Equation Solver Data Preparation**

## Description

This Jupyter notebook handles the crucial first steps in building an image-based equation solver: data ingestion and feature engineering. It automates the process of downloading a compressed image dataset, extracting its contents, and then processing individual images. The images, which represent handwritten digits (0-9) and basic arithmetic operators (`+`, `-`, `*`, `/`), are resized, converted to grayscale, flattened, normalized, and finally compiled into a structured Pandas DataFrame and saved as a CSV file. This prepared dataset will serve as input for training a classification model capable of recognizing mathematical symbols.

## Features

-   **Automated Data Ingestion**: Downloads and unzips the raw image dataset directly within the notebook environment.
-   **Image Preprocessing**:
    -   Loads images in grayscale.
    -   Resizes all images to a consistent 32x32 pixel dimension.
    -   Flattens image pixel data into a 1D array.
    -   Normalizes pixel values to a 0-1 range.
-   **Dynamic Labeling**: Extracts labels for each image based on its directory structure (e.g., images in the '0' folder are labeled '0').
-   **Data Shuffling**: Randomly shuffles the dataset to ensure a good mix of classes.
-   **CSV Export**: Saves the processed image data and their corresponding labels into a `data.csv` file, ready for machine learning.
-   **Image Visualization**: Includes utilities to display sample images with their labels, aiding in data understanding and verification.

## Installation

To set up and run this notebook, follow these steps:

1.  **Clone the repository** (or download the `Equation_Solver.ipynb` file).
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Recommended Environment**: This notebook is designed to run seamlessly in **Google Colab**. No local environment setup is strictly necessary if using Colab.

3.  **Local Environment (if not using Colab)**:
    If you prefer to run it locally, you'll need:
    -   Python 3.x
    -   Jupyter Notebook or JupyterLab

    Install the required Python libraries using pip:
    ```bash
    pip install opencv-python numpy pandas matplotlib
    ```

## Usage

You can run the notebook in Google Colab or your local Jupyter environment.

### Running in Google Colab

1.  Open Google Colab in your browser.
2.  Click `File > Upload notebook` and upload `Equation_Solver.ipynb`.
3.  Run all cells (`Runtime > Run all`). The notebook will automatically download the dataset, process it, and save `data.csv` in your Colab environment.

### Running Locally (Jupyter)

1.  Navigate to the directory containing `Equation_Solver.ipynb` in your terminal.
2.  Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open `Equation_Solver.ipynb` from the Jupyter interface.
4.  Execute each cell sequentially.

#### Key Code Snippets

**1. Data Ingestion:**
This cell downloads and extracts the dataset.
```python
!wget https://cainvas-static.s3.amazonaws.com/media/user_data/Yuvnish17/data.zip
!unzip -qo data.zip
```

**2. Extracting and Preprocessing Images:**
The `extract_images_np` function loads, resizes, flattens, and shuffles the images.
```python
def extract_images_np(dir):
  images=[]
  labels=[]

  for folder in os.listdir(dir):
    path=os.path.join(dir,folder)
    for image in os.listdir(path):
      img=cv2.imread(os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
      img=cv2.resize(img,(32,32))
      img=img.flatten()
      images.append(img)
      labels.append(folder)
  data=list(zip(images,labels))
  random.shuffle(data)

  images,labels=zip(*data)

  return np.array(images),np.array(labels)

data_dir="/content/data/dataset" # Adjust if running locally and data is elsewhere
images,labels=extract_images_np(data_dir)
```

**3. Normalization and CSV Export:**
Images are normalized and saved to a CSV file.
```python
import pandas as pd
images=np.array(images)
images=images/255.0
df_X=pd.DataFrame(images)
df_X['label']=labels
df_X.to_csv('data.csv',index=False)
```

**4. Visualizing Sample Images:**
This code block displays a grid of random images from each class (0-9, add, sub, mul, div).
```python
import matplotlib.pyplot as plt

folders = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'sub', 'mul', 'div']
images = []

for folder in folders:
    folder_path = '/content/data/dataset/' + folder # Adjust path if running locally
    image_files = os.listdir(folder_path)
    if image_files:
        random_image_file = random.choice(image_files)
        image = cv2.imread(os.path.join(folder_path, random_image_file), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (32, 32))
        images.append(image)

images = np.array(images)
images = images / 255.0
images = np.expand_dims(images, axis=-1)

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(images):
        ax.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title(f"{folders[i]}")
        ax.axis('off')
    else:
        ax.axis('off')
plt.tight_layout()
plt.show()
```

## Dependencies

-   Python 3.x
-   `opencv-python` (for image loading and processing)
-   `numpy` (for numerical operations)
-   `pandas` (for data manipulation and CSV export)
-   `matplotlib` (for plotting and visualization)

These can be installed via `pip` as shown in the [Installation](#installation) section.

## Project Status

This notebook successfully completes the data preparation and feature engineering phase of the Equation Solver project. The output `data.csv` provides a clean, preprocessed dataset ready for machine learning model training.

**Future Work:**
-   Develop and train a Convolutional Neural Network (CNN) or other suitable model for character recognition.
-   Implement an algorithm to parse a sequence of recognized symbols into a mathematical expression.
-   Create logic to solve the parsed mathematical expression.
-   Build a user interface for inputting equations (e.g., drawing, uploading an image).