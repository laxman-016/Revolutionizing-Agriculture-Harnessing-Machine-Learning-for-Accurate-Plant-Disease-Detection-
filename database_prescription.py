import mysql.connector
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

class_labels = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy', 'plantvillage']  # Replace with your class labels

# Load your pre-trained CNN model
model = tf.keras.models.load_model(r'C:\Users\laxma\OneDrive\Desktop\project_folder\your_model_folder\leaf_disease_detection_model.h5')
# Connect to the MySQL database
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="****",
    database="mydata"
)

cursor = connection.cursor()

# Define a function to identify crop diseases and retrieve medicine
def identify_crop_disease(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    disease_name = class_labels[predicted_class_index]

    return disease_name

# Define a function to retrieve medicine from the database
def get_medicine_for_disease(disease_name):
    query = "SELECT medicine FROM prescription WHERE disease = %s"
    cursor.execute(query, (disease_name,))
    result = cursor.fetchone()
    if result:
        medicine = result[0]
        return medicine
    else:
        return None  # Disease not found in the database

# Example usage
image_path = r'C:\Users\laxma\.vscode\CNN Model\p2.jpg'

try:
    disease = identify_crop_disease(image_path)
    medicine = get_medicine_for_disease(disease)

    if medicine:
        print(f'\nIdentified Disease : {disease}\n')
        print(f'Medicine for {disease}:\n')
        # Add line breaks within the medicine description
        medicine_with_line_breaks = medicine.replace('. ', '.\n')
        print(medicine_with_line_breaks)
    else:
        print(f'Disease not found in the database.')

except Exception as e:
    print(f'Error: {str(e)}')

# Close the database connection when done
cursor.close()
connection.close()

