import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd


print('getcwd:      ', os.getcwd())
# filename=(r"..\src\features\PotatoEarlyBlight1.JPG")
# print('le filename de img_color est :', filename)
filename="c:\\Users\\Maria PaulaValentina\\Desktop\\DataScience\\GIT\\JAN24_PLANT_RECOGNITION\\src\\features\\TomatoEarlyBlight4.JPG"
where= "c:\\Users\Maria PaulaValentina\\Desktop\\DataScience\\Proyecto\\Etape22\\Etape 2 - Maladie_sur_les_plantes\\01_New_Plant_Diseases_Dataset\\01_New_Plant_Diseases_Dataset"
where_shape= "c:\\Users\\Maria PaulaValentina\\Desktop\\DataScience\\Proyecto\\Etape1\\Etape11\\Only_Healthy\\New_Plant_Diseases_Dataset_only_healthy\\shape_2"
where_shape_original= "c:\\Users\\Maria PaulaValentina\\Desktop\\DataScience\\Proyecto\\Etape1\\Etape11\\Only_Healthy\\New_Plant_Diseases_Dataset_only_healthy\\shape_original"
where_s= "c:\\Users\\Maria PaulaValentina\\Desktop\\DataScience\\Proyecto\\Etape1\\Etape11\\Only_Healthy\\New_Plant_Diseases_Dataset_only_healthy\\HSV_S"
where_H= "c:\\Users\\Maria PaulaValentina\\Desktop\\DataScience\\Proyecto\\Etape1\\Etape11\\Only_Healthy\\New_Plant_Diseases_Dataset_only_healthy\\HSV_H"
where_RGB= "c:\\Users\\Maria PaulaValentina\\Desktop\\DataScience\\Proyecto\\Etape1\\Etape11\\Only_Healthy\\New_Plant_Diseases_Dataset_only_healthy\\RGB"
where_contour= "c:\\Users\\Maria PaulaValentina\\Desktop\\DataScience\\Proyecto\\Etape1\\Etape11\\Only_Healthy\\New_Plant_Diseases_Dataset_only_healthy\\withContour"
where_rect_ellipse= "c:\\Users\\Maria PaulaValentina\\Desktop\\DataScience\\Proyecto\\Etape1\\Etape11\\Only_Healthy\\New_Plant_Diseases_Dataset_only_healthy\\rect_ellipse"
save_csv=  "c:\\Users\\Maria PaulaValentina\\Desktop\\DataScience\\Proyecto\\Etape1\\Etape11\\Only_Healthy\\New_Plant_Diseases_Dataset_only_healthy\\Nueva carpeta"
# dir_train=os.listdir(where_train)

type_data=["valid","train"]
theClass=['Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 
          'Corn_(maize)___healthy', 'Grape___healthy', 'Peach___healthy', 
          'Pepper,_bell___healthy', 'Potato___healthy', 'Raspberry___healthy', 
          'Soybean___healthy', 'Strawberry___healthy', 'Tomato___healthy']

feat=['name',"area","perimeter","isConvex","aspect_ratio","extent","solidity","eccentricity"]

data=[]

for i in type_data:

    

    where_data= where+"\\"+i

    which_class=os.listdir(where_data)

    for l in which_class:

        print(l)
        folder_class= where_data+"\\"+l

        which_pic=os.listdir(folder_class)

        # fig = plt.figure(figsize = (12,12))
        # plt.title(i) 
    
    

        for j in range(0,len(which_pic)):
        # for j in range(0,3):

            pic=folder_class+"\\"+which_pic[j]
        
            
            plant_color = cv2.imread(pic, cv2.IMREAD_COLOR)
            # plant_colorv2 = cv2.cvtColor(plant_color, cv2.COLOR_BGR2RGB)
            plant_colorv3 = cv2.cvtColor(plant_color, cv2.COLOR_BGR2HSV)

            # fig.add_subplot(3,3,j+1)
            plant_colorv3_filtreG = cv2.GaussianBlur(plant_colorv3,(5,5),0)

            mask_green = cv2.inRange(plant_colorv3, (36,0,0), (86,255,255))
            # find the brown color
            mask_brown = cv2.inRange(plant_colorv3, (8, 60, 20), (30, 255, 200))
            # find the yellow color in the leaf
            mask_yellow = cv2.inRange(plant_colorv3, (21, 39, 64), (40, 255, 255))

            # find any of the three colors(green or brown or yellow) in the image
            mask = cv2.bitwise_or(mask_green, mask_brown)
            mask = cv2.bitwise_or(mask, mask_yellow)

            # Bitwise-AND mask and original image
            res = cv2.bitwise_and(plant_color,plant_color, mask= mask)

            inv_mask = cv2.threshold(mask, 0, 255,cv2.THRESH_BINARY_INV)[1]

            contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            

            maxArea=0
            c=0

            for k in range(0,len(contours)):
                area = cv2.contourArea(contours[k])
                if area >= maxArea:
                    maxArea = area
                    c = k

            # print("Contour: ",c)

            if len(contours)!=0:

                cv2.drawContours(res, contours, c, (0, 255, 0), 3) 
                
                area = cv2.contourArea(contours[c])
                perimeter = cv2.arcLength(contours[c],True)
                isConvex = cv2.isContourConvex(contours[c])

                x,y,w,h = cv2.boundingRect(contours[c])
                img = cv2.rectangle(res,(x,y),(x+w,y+h),(255,0,0),2)

                aspect_ratio = float(w)/h

                rect_area = w*h
                extent = float(area)/rect_area


                hull = cv2.convexHull(contours[c])
                hull_area = cv2.contourArea(hull)
                solidity = float(area)/hull_area

                ellipse = cv2.fitEllipse(contours[c])
                image = cv2.ellipse(res, ellipse, (255,0, 255), 2, cv2.LINE_AA)
            
            

                ma=min(ellipse[1])
                MA=max(ellipse[1])

                ecc= np.sqrt(1-(ma/MA)**2)
            else:
                area=float("NaN")
                perimeter=float("NaN")
                isConvex=float("NaN")
                aspect_ratio=float("NaN")
                extent=float("NaN")
                solidity=float("NaN")
                ecc=float("NaN")

            # im =np.array(type_data)== i
            # info_class=np.array(theClass)== l
            # im=list(im) + list(info_class) 

            im=[i,l]


            im=im+[pic,area, perimeter,isConvex, aspect_ratio,
             extent, solidity, ecc]
            
            if len(im) !=10:
                print(im)
                break

            data.append(im)
          

            # plt.imshow(image)

            # plt.imshow(inv_mask,cmap='gray')
            
            # plt.xticks([])
            # plt.yticks([])


        
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    # plt.close()
    # fig.savefig(where_rect_ellipse+"\\"+i+'.png')


the_columns = ["type_data","theClass"] + feat

df= pd.DataFrame(np.array(data),columns=the_columns)

df.to_csv(save_csv+"\\feat_shape2all.csv")

print("Done")

# df.to_excel("feat_shape.xlsx")
