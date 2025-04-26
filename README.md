# NeuroScan-AI
Deep learning-powered early detection of Alzheimer's disease using MRI brain scans with explainable AI (Grad-CAM) techniques.

Project Overview
NeuroScan-AI leverages deep learning techniques to automatically detect Alzheimer's disease from MRI scans. The model uses a Convolutional Neural Network (CNN) to classify brain scans into different stages of Alzheimer's disease: Non-Demented (ND), Mild Demented (MD), Moderate Demented (MOD), and Very Mild Demented (VMD).

The project integrates Explainable AI (XAI) using Grad-CAM to visualize the important brain regions the model uses for classification, improving clinical trust and transparency.

🗂️ Project Structure
data/ — MRI images categorized by condition (e.g., NonDemented, MildDemented, ModerateDemented, VeryMildDemented)

notebooks/ — Jupyter notebooks for training, evaluation, and experimentation

models/ — Saved model files and weights

outputs/ — Visual outputs, including confusion matrices and Grad-CAM heatmaps

app/ — Streamlit app for live predictions (optional)

README.md — Project description and instructions

requirements.txt — List of required Python libraries

📈 Technologies Used
Python 3.8+

TensorFlow / Keras for model development

OpenCV for image preprocessing

NumPy, Pandas for data manipulation

Scikit-learn for model evaluation

Matplotlib, Seaborn for data visualization

Streamlit for optional deployment

🧠 Model Architecture
Input Layer: MRI scans resized to a standard size (e.g., 224x224)

Convolutional Layers: Multiple Conv2D layers to extract features like edges and textures

Pooling Layers: MaxPooling to reduce spatial dimensions

Fully Connected Layers: Dense layers to process extracted features and output classification

Dropout Layers: Dropout for regularization

Output Layer: Softmax activation for classification into Alzheimer's stages

🔬 How to Run the Project
Clone the repository:

git clone https://github.com/yourusername/NeuroScan-AI.git
cd NeuroScan-AI
Install dependencies:


pip install -r requirements.txt
Download the Alzheimer's MRI dataset and place it inside the data/ folder. You can find the dataset on Kaggle (e.g., Alzheimer's Disease Neuroimaging Initiative dataset).

Run the notebook to train and evaluate the model:


jupyter notebook notebooks/Alzheimer's_Detection.ipynb
(Optional) Launch the Streamlit app for live predictions:


streamlit run app/app.py
Results
Training Accuracy: ~95%+

Validation Accuracy: ~92%+

AUC Score: ~0.96

You can find confusion matrices and Grad-CAM visualizations in the outputs/ folder.

🚀 Future Enhancements
3D-CNN: Implement 3D Convolutional Neural Networks for volumetric MRI analysis.

Multi-modal Learning: Integrate additional data sources (e.g., patient demographics, biomarkers) to enhance model performance.

Real-time Detection: Develop the system as a real-time clinical tool for hospitals to perform Alzheimer's screening.

🙏 Acknowledgements
Alzheimer's MRI Dataset - Kaggle

TensorFlow and Keras teams

Research papers on Alzheimer's early detection

📬 Contact
Created by Aadhira Suleim A R — AI Healthcare Engineer
Feel free to connect with me on LinkedIn.

Let's revolutionize healthcare with AI and early Alzheimer's detection! 🚀
