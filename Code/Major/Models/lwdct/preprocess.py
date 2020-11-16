try:
    import cv2 as cv2
    import os
    import pickle
    import numpy as np
    print("Libraries Loaded Successfully!!!")
    print(os.getcwd())

except:
    print("Library not Found ! ")



#Helper Function to get_labels
# Each Video is labeled using an naming convention VideoNAME_sEQUENCENumber_ClassLabel.avi // Frame Number is appended at the end "frameX"

def get_label(img):
    x=img[(img.rindex("_"))-1]
    return int(x)

# Helper Function to read images and labels ,resize and store in a matrix


def process_frames(PATH,IMAGE_SIZE):
    # List Variables

    image_data = []
    x_data = []
    y_data = []

    for im in os.listdir(PATH):
        path = os.path.join(PATH, im)
        class_label = get_label(im)

        try:  # if any image is corrupted
            image_temp = cv2.imread(path)  # Read Image as numbers
            image_temp_resize = cv2.resize(image_temp, (IMAGE_SIZE, IMAGE_SIZE))
            image_data.append([image_temp_resize])
            y_data.append(class_label)
        except:
            pass
    data = np.asanyarray(image_data)
    # Iterate over the Data
    for x in data:
        x_data.append(x[0])  # Get the X_Data

    X_Data = np.asarray(x_data)  # Normalize Data
    Y_Data =np.asarray(y_data)


    return X_Data,Y_Data

def pickle_imageval(PATH,IMAGE_SIZE):
    """
    :return: None, Creates a Pickle Object of DataSet into CWD
    """
    # Call the Function and Get the Data
    X_Data,Y_Data = process_frames(PATH,IMAGE_SIZE)
    # Write the Entire Data into a Pickle File
    pickle_out = open('Xval_Data','wb')
    pickle.dump(X_Data, pickle_out)
    pickle_out.close()
    # Write the Y Label Data
    pickle_out = open('Yval_Data', 'wb')
    pickle.dump(Y_Data, pickle_out)
    pickle_out.close()
    print("Pickled Images Successfully at ", os.getcwd())
    return X_Data,Y_Data

def load_val(PATH,IMAGE_SIZE):
    try:
        # Read the Data from Pickle Object
        X_Temp = open('Xval_Data', 'rb')
        X_Data = pickle.load(X_Temp)

        Y_Temp = open('Yval_Data', 'rb')
        Y_Data = pickle.load(Y_Temp)

        print('Reading Dataset from Pickle Object in ',os.getcwd())

        return X_Data, Y_Data

    except:
        print('Could not Found Pickle File ')
        print('Loading File and Dataset  ..........')
        X_Data, Y_Data = pickle_imageval(PATH , IMAGE_SIZE)
        return X_Data, Y_Data

def pickle_image(PATH,IMAGE_SIZE):

        """
        :return: None, Creates a Pickle Object of DataSet into CWD
        """
        # Call the Function and Get the Data
        X_Data,Y_Data = process_frames(PATH,IMAGE_SIZE)

        # Write the Entire Data into a Pickle File
        pickle_out = open('X_Data','wb')
        pickle.dump(X_Data, pickle_out)
        pickle_out.close()

        # Write the Y Label Data
        pickle_out = open('Y_Data', 'wb')
        pickle.dump(Y_Data, pickle_out)
        pickle_out.close()

        print("Pickled Images Successfully at ", os.getcwd())
        return X_Data,Y_Data

def load_dataset(PATH , IMAGE_SIZE):
    try:
        # Read the Data from Pickle Object
        X_Temp = open('X_Data', 'rb')
        X_Data = pickle.load(X_Temp)

        Y_Temp = open('Y_Data', 'rb')
        Y_Data = pickle.load(Y_Temp)

        print('Reading Dataset from Pickle Object in ',os.getcwd())

        return X_Data, Y_Data

    except:
        print('Could not Found Pickle File ')
        print('Loading File and Dataset  ..........')
        X_Data, Y_Data = pickle_image(PATH , IMAGE_SIZE)
        return X_Data, Y_Data




# Generate 20 frames for each video in
def frames_20(n, pathv,patho):
    lists = os.listdir(pathv)

    for vid in lists:
        vidcap = cv2.VideoCapture(pathv + vid)
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        frames_step = total_frames // n
        for i in range(n):
            # here, we set the parameter 1 which is the frame number to the frame (i*frames_step)
            vidcap.set(1, i * frames_step)
            success, image = vidcap.read()
            resize = cv2.resize(image, (64, 64))
            # save your image
            st, x = vid.split(".")
            a,b,c = st.split("_")
            fc=str(i)
            tmp=[str(a),str(b),fc,str(c)]
            strg ='_'.join(tmp)
            strg=strg+".jpg"
            cv2.imwrite(patho+strg, resize)
            print(".", end=' ')
        vidcap.release()



def load_test(path):
    dat = os.listdir(path)
    img_data_list = []
    l=[]

    for im in dat:
        l.append(get_label(im))
        inp = cv2.imread(path + im)
        img_data_list.append(inp)

    # img_data_list #list shape is [ n_samples x [64 x 64 ] ]
    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255

    return img_data,l

def testok():
    print("yea wassap")

