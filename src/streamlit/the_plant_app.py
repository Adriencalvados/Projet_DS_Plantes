import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import streamlit as st
import tensorflow as tf
from keras.models import model_from_json
from keras.preprocessing import image as image_utils
from tensorflow.keras.applications import VGG19
from keras import layers, models, optimizers, losses, metrics,callbacks,utils,Model,Sequential,Input 
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
#import imageio.v3 as iio
import sys, subprocess
import shutil, shap
import seaborn as sns
import json
import requests

st.set_page_config(page_title='🌿🌱 Reco Plantes 🌿🌱' , layout='centered')
#st.image(["dataLog.jpg","LogomINES.png"],width=300 )
#st.image([f"src/features/dataLog.jpg",f"src/features/LogomINES.png"],width=300 )

st.title(" 🌱 Reconnaissance de plantes 🌿")
st.sidebar.image([f"src/features/dataLog.jpg"],width=200)
st.sidebar.title("Sommaire")
pages=["Introduction","DataVizualization", "Modélisation", "Interprétations"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.image([f"src/features/LogomINES.png"],width=200)
st.sidebar.write("Adrien Pinel")
st.sidebar.write("Guillaume Lezan")
st.sidebar.write("María Flechas")

#////////////////////////////////////////////Load////////////////////////////////////

logged_model = "models/model_pour_interpretation/e353de44e64042d3b9da44c9f1768402/artifacts/model/data/model.keras"
@st.cache_data
def pull_data_with_dvc():
      cmd = [sys.executable, "-m", "dvc", "pull"]
      result = subprocess.run(cmd, capture_output=True, text=True)
      if result.returncode == 0:
          #shutil.unpack_archive("models/model_retenu.zip", "models/model_retenu")
          shutil.unpack_archive("models/model_pour_interpretation.zip", "models/model_pour_interpretation")
          
      else:
          st.write("Error pulling data!")
          st.write(result.stderr)
def modifjson(file_path) :
    with open(file_path, 'r') as f:
        data = json.load(f)
    # 2. Modifier les valeurs des clés
    data['private_key_id'] = st.secrets["credentials"]["private_key_id"]
    data['private_key'] = st.secrets["credentials"]["private_key"]
    # 3. Écrire (écraser) le fichier JSON avec les nouvelles données
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)    
  # Function to convert image to base64
def image_to_base64(im):
      # img = Image.open(img_path)
      buffered = BytesIO()
      im.save(buffered, format="PNG")
      return base64.b64encode(buffered.getvalue()).decode()

  # Function to generate HTML for the image
def path_to_image_html(path):
      return f'<img src="data:image/png;base64,{image_to_base64(path)}" width="300" >'

@st.cache_resource  # 👈 Add the caching decorator
def load_trained_model(logged_model):
    base_model = VGG19(include_top=False,weights=None,input_shape=(256,256,3))
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(2048, activation='relu')(x)
    x= layers.Dropout(rate=0.2)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x= layers.Dropout(rate=0.2)(x)
    output = layers.Dense(38, activation='softmax')(x)

    cnn = models.Model(inputs=[base_model.input], outputs=[output])
    cnn.compile(optimizer=optimizers.Adam(learning_rate=1e-8),loss='categorical_crossentropy',metrics=['accuracy'])
    cnn.load_weights(logged_model)
    cnn.summary()
    return cnn
    
def my_preprocessing_func(img):
    image = np.array(img)
    return image / 255

def pre_process_img_streamlit(upload_file):
    # image_name = "/content/"+upload_file[i].name
    img = Image.open(upload_file)
    image_array = np.array(img)
    # im=cv2.cvtColor(cv2.imread(image_array),cv2.COLOR_BGR2RGB)
    im=cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
    im=cv2.resize(im,(256,256))
    im=my_preprocessing_func(im)
    print("shape",im.shape)
    d=im.reshape(1,256,256,3)
    print("reshape",d.shape)
    return d,img

@st.cache_data
def charger_dataframes():
    df=pd.read_csv("references/df_area.csv",sep=",",index_col=0,nrows=90000) 
    labels=pd.read_csv("references/labels.csv",header=0,index_col=0)
    return df,labels
#////////////////////////////////////////////FIN Load////////////////////////////////////
with st.spinner('Wait for load...'):    
    modifjson('models/atomic-graph-437912-e3-006877ce0826.json')
    pull_data_with_dvc()
    df,labels=charger_dataframes()


#//////////////////////////////////////////pages[0]
if page == pages[0]:
    st.header("Introduction")
    st.markdown("""
    ### 🌿 **Appliquer les algorithmes de Deep Learning pour** :

    - 🔍 **Détecter à partir d'une photo de plante :**
    - 📸 L’**espèce** à laquelle cette plante appartient
    - 🦠 Si la plante est **porteuse ou non d’une maladie**

    ---
    ### 🚀 **Application** :
    Une fois **entraîné**, le modèle pourra traiter **n’importe quelle image de plante** prise au moyen d’un simple **appareil photo** 📱📷.

    ---
    """)
    st.markdown("Multi plants &mdash;:tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:")
    st.image(Image.open("reports/figures/2-1-0-Adrien_1ervisu.png"))
#//////////////////////////////////////////pages[1]
if page == pages[1]:
    st.header("Exploration des images")
    # Texte avec des smileys et du markdown
    markdown_text = """
    **🌱 Source** : La base de données **Kaggle** sur les maladies des plantes.

    - L'échantillon comprend **87 867** images réparties en **38 classes** 🌿, avec une moyenne de **2312 images** par classe 📸.
    
    - La **variable cible** correspond au nom du dossier lié à une plante 🪴, parfois agrégé avec le nom de la maladie 🌡️ : nous sommes dans un système **supervisé** de type **classification** 🧠.

    - 🔄 La présence d'images similaires avec différentes **rotations**, **luminosité**, et **déformations** indique qu'une étape de **préprocessing** a été appliquée pour **augmenter** l'échantillon 🔧.
    """

    # Affichage en markdown dans Streamlit
    st.markdown(markdown_text)
    st.subheader("Describe")
    st.dataframe(df.describe())
    fig = plt.figure()
    sns.countplot(x = 'type_data', data = df)
    plt.title("Répartition des images")
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.countplot(x = 'theClass', data = df)
    plt.title('Class distribution in training & valid')
    plt.xticks(rotation='vertical')
    plt.xlabel("Nom de la classe")
    plt.ylabel("Nombre d'éléments")
    plt.legend()
    st.pyplot(fig)
    
    fig = plt.figure()
    sns.boxplot(data=df,x='theClass',y="area")
    plt.xticks(rotation=90);
    plt.title("Train data: Area")
    st.pyplot(fig)
    
#//////////////////////////////////////////pages[2]
if page==pages[2]:
    with st.spinner('Wait for load...'):
        cnn=load_trained_model(logged_model)

    st.header("Sélecteur d'acquisition d'image")

    # Options pour l'acquisition d'image
    option = st.selectbox(
        "Choisissez la méthode d'acquisition d'image",
        ("Acquisition par webcam", "Upload d'une image", "Extraction d'image","Exemple d'images")
    )
    
    if option == "Acquisition par webcam":
        st.write("Cliquez sur le bouton ci-dessous pour capturer une image avec votre webcam.")

        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            d,img=pre_process_img_streamlit(img_file_buffer)    
            pred=cnn.predict(d)
            predicted_class_indices=np.argmax(pred,axis=1)
            print("Class index:",predicted_class_indices[0])
            the_class= labels.iloc[predicted_class_indices[0]][0]
            probaWeb = np.round(pred[0][predicted_class_indices[0]]*100,1)
            st.write("Prédictions : "+the_class+" avec "+ str(probaWeb)+'%'+" de probabilité.")
            
     # 2. Upload d'une image
    elif option == "Upload d'une image":
        upload_file = st.file_uploader("Télecharger des images de plantes!",accept_multiple_files=True)
        print("upload_file :",upload_file)

        if upload_file!= None:

            images_name=[]
            the_classes=[]
            files_names=[]
            probabilite = []
            for i in range(0,len(upload_file)):
                        
                d,img=pre_process_img_streamlit(upload_file[i])    
                pred=cnn.predict(d)
                predicted_class_indices=np.argmax(pred,axis=1)
                print("Class index:",predicted_class_indices[0])
                the_class= labels.iloc[predicted_class_indices[0]][0]
                print("Class :",the_class)
                probabilite.append(np.round(pred[0][predicted_class_indices[0]]*100,1))
                the_classes.append(the_class)
                files_names.append(upload_file[i].name)
                images_name.append(img)
            # Sample data
            data = {
                "Classification": the_classes,
                "Nom du ficher ":files_names,
                "Probabilité % ":probabilite,
                "Image":images_name    
            }

            df = pd.DataFrame(data)

            # Convert the image path column to actual images using HTML
            df['Image'] = df['Image'].apply(path_to_image_html)

            # Display the dataframe with images in Streamlit
            st.write(df.to_html(escape=False), unsafe_allow_html=True)
            # st.dataframe(df)
        
    # 3. Extraction d'image depuis un lien
    elif option == "Extraction d'image":
        url=f"https://photoshop-kopona.com/uploads/posts/2018-06/1528827395_strawberry-92.jpg"
        img_url = st.text_input("URL",url)

        if img_url:
            try:
                response = requests.get(img_url)
                img = Image.open(BytesIO(response.content))
                image_array = np.array(img)
                im=cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
                im=cv2.resize(im,(256,256))
                im=my_preprocessing_func(im)
                print("shape",im.shape)
                d=im.reshape(1,256,256,3)   
                pred=cnn.predict(d)
                predicted_class_indices=np.argmax(pred,axis=1)
                print("Class index:",predicted_class_indices[0])
                the_class= labels.iloc[predicted_class_indices[0]][0]
                probaExtract = np.round(pred[0][predicted_class_indices[0]]*100,1)
                st.write("Prédictions : "+the_class+" avec "+ str(probaExtract)+'%'+" de probabilité.")
                st.image(img, caption="Image extraite depuis l'URL")
            except Exception as e:
                st.error(f"Erreur lors de l'extraction de l'image : {e}")

    # 4. Exemple d'image venant du repo
    elif option == "Exemple d'images":
        st.write("Sélectionnez un exemple d'image à partir du repo.")

        # Liste des exemples d'images du repo (ex: stockées localement)
        example_images = {
            "AppleCedarRust1": "src/features/test/AppleCedarRust1.JPG",
            "AppleCedarRust2": "src/features/test/AppleCedarRust2.JPG",
            "AppleCedarRust3": "src/features/test/AppleCedarRust3.JPG",
            "AppleCedarRust4": "src/features/test/AppleCedarRust4.JPG",
            "AppleScab1": "src/features/test/AppleScab1.JPG",
            "AppleScab2": "src/features/test/AppleScab2.JPG",
            "AppleScab3": "src/features/test/AppleScab3.JPG",
            "CornCommonRust1": "src/features/test/CornCommonRust1.JPG",
            "CornCommonRust2": "src/features/test/CornCommonRust2.JPG",
            "CornCommonRust3": "src/features/test/CornCommonRust3.JPG",
            "PotatoEarlyBlight1": "src/features/test/PotatoEarlyBlight1.JPG",
            "PotatoEarlyBlight2": "src/features/test/PotatoEarlyBlight2.JPG",
            "PotatoEarlyBlight3": "src/features/test/PotatoEarlyBlight3.JPG",
            "PotatoEarlyBlight4": "src/features/test/PotatoEarlyBlight4.JPG",
            "PotatoEarlyBlight5": "src/features/test/PotatoEarlyBlight5.JPG",
            "PotatoHealthy1": "src/features/test/PotatoHealthy1.JPG",
            "PotatoHealthy2": "src/features/test/PotatoHealthy2.JPG",
            "TomatoEarlyBlight1": "src/features/test/TomatoEarlyBlight1.JPG",
            "TomatoEarlyBlight2": "src/features/test/TomatoEarlyBlight2.JPG",
            "TomatoEarlyBlight3": "src/features/test/TomatoEarlyBlight3.JPG",
            "TomatoEarlyBlight4": "src/features/test/TomatoEarlyBlight4.JPG",
            "TomatoEarlyBlight5": "src/features/test/TomatoEarlyBlight5.JPG",
            "TomatoEarlyBlight6": "src/features/test/TomatoEarlyBlight6.JPG",
            "TomatoHealthy1": "src/features/test/TomatoHealthy1.JPG",
            "TomatoHealthy2": "src/features/test/TomatoHealthy2.JPG",
            "TomatoHealthy3": "src/features/test/TomatoHealthy3.JPG",
            "TomatoHealthy4": "src/features/test/TomatoHealthy4.JPG",
            "TomatoYellowCurlVirus1": "src/features/test/TomatoYellowCurlVirus1.JPG",
            "TomatoYellowCurlVirus2": "src/features/test/TomatoYellowCurlVirus2.JPG",
            "TomatoYellowCurlVirus3": "src/features/test/TomatoYellowCurlVirus3.JPG",
            "TomatoYellowCurlVirus4": "src/features/test/TomatoYellowCurlVirus4.JPG",
            "TomatoYellowCurlVirus5": "src/features/test/TomatoYellowCurlVirus5.JPG",
            "TomatoYellowCurlVirus6": "src/features/test/TomatoYellowCurlVirus6.JPG",            
        }

        # Choisir une image d'exemple
        example_choice = st.selectbox("Choisissez une image d'exemple", list(example_images.keys()))

        if example_choice:
            # Charger l'image sélectionnée à partir du répertoire local
            img_path = example_images[example_choice]
            
            try:
                image = Image.open(img_path)
                
                d,img=pre_process_img_streamlit(img_path)    
                pred=cnn.predict(d)
                predicted_class_indices=np.argmax(pred,axis=1)
                print("Class index:",predicted_class_indices[0])
                the_class= labels.iloc[predicted_class_indices[0]][0]
                probaRepo = np.round(pred[0][predicted_class_indices[0]]*100,1)
                st.write("Prédictions : "+the_class+" avec "+ str(probaRepo)+'%'+" de probabilité.")  
                st.image(image, caption=f"Exemple d'image : {example_choice}")        
            except FileNotFoundError:
                st.error(f"Erreur : L'image '{example_choice}' n'a pas été trouvée dans le répertoire.")
    markdown_text = """
    ### Modélisation - Réseaux de Neurones Convolutifs (CNN) 🧠🔍

    - **Plusieurs modèles CNN testés** : VGG16, VGG19, ResNet 🤖
    - **Métrique principale** : Accuracy (95% pour le meilleur modèle) 📊✅
    - **Modèle final** : VGG19 avec des couches convolutives pour extraire les caractéristiques visuelles 🎯🖼️
                    """
    st.markdown(markdown_text)           
#//////////////////////////////////////////pages[3]
if page == pages[3]:
    with st.spinner('Wait for load...'):
        cnn=load_trained_model(logged_model)

    st.header("Interprétation")
    upload_file = st.file_uploader("Télecharger des images de plantes!",accept_multiple_files=False)
    print("upload_file :",upload_file)
    if upload_file!= None:
        with st.spinner('Wait for load...'):
            img = Image.open(upload_file)
            image_array = np.array(img)
            im=cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
            im=cv2.resize(im,(256,256))
            im=my_preprocessing_func(im)
            #"reshape"
            d=im.reshape(1,256,256,3)

            markdown_text = """
            ### 🌟 Grad-CAM
            Grad-CAM (Gradient-weighted Class Activation Mapping) est une méthode qui met en lumière les zones importantes 
            d'une image pour la décision d'un modèle.  
            """
            st.markdown(markdown_text)  
            last_conv_layer = cnn.get_layer("block5_conv4")
            last_conv_layer_model = Model(cnn.inputs, last_conv_layer.output)
            classifier_input = Input(shape=last_conv_layer.output.shape[1:])
            x = classifier_input
            layer_names = []
            for layer in cnn.layers[21:]:
                layer_names.append(layer.name) # Noms des couches, afin que vous puissiez les intégrer à votre graphique
            print(layer_names)
            for layer_name in layer_names:
                x = cnn.get_layer(layer_name)(x)
            classifier_model = Model(classifier_input, x)
            with tf.GradientTape() as tape:
                inputs = d
                last_conv_layer_output = last_conv_layer_model(inputs)
                tape.watch(last_conv_layer_output)
                preds = classifier_model(last_conv_layer_output)
                top_pred_index = tf.argmax(preds[0])
                top_class_channel = preds[:, top_pred_index]
            grads = tape.gradient(top_class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            last_conv_layer_output = last_conv_layer_output.numpy()[0]
            pooled_grads = pooled_grads.numpy()
            for i in range(pooled_grads.shape[-1]):
                last_conv_layer_output[:, :, i] *= pooled_grads[i]
            # Average over all the filters to get a single 2D array
            gradcam = np.mean(last_conv_layer_output, axis=-1)
            # Clip the values (equivalent to applying ReLU)
            # and then normalise the values
            gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
            gradcam = cv2.resize(gradcam, (256, 256))

        fig=plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(d[0])
        plt.title("Image originale")
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(gradcam)
        plt.title("Filtre CNN")
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(d[0])
        plt.imshow(gradcam, alpha=0.5)
        plt.title('Interprétabilité des zones de recherche')
        plt.axis('off')
        st.pyplot(fig)
                    






