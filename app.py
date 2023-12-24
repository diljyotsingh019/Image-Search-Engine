from utils import * # Used to import necessary functions
from imutils import paths #Used to list images from a given directory
from werkzeug.utils import secure_filename # Used to fetch a filename from Flask request
from config import * # Used to import necessary variables
from logs import * #Used to manage logs
import os #Used to manage operarting system operations
import time # Used to manage time
from flask import Flask, Response, redirect, url_for, render_template, request # Used for web app deployments


logger = log_setup("main", LOGFILE) #Initialising logger object to start logging

imagepaths = list(paths.list_images(IMAGEPATHS)) #Creating a list of all the images in the directory
data = []
t1 = time.time()
logger.debug("[INFO] Reading Images.....")
for i, j in enumerate(imagepaths): #Reading all the images and generating histograms
    image = cv2.imread(j)
    data.append(feature_extraction(image))


logger.debug("[INFO] Histograms calculated successfully, Time Taken: {:.2f}seconds".format(time.time() - t1))
data = np.array(data)

##############################Deployment#########################################
app = Flask(__name__) #Initialising flask object

@app.route("/", methods = ["POST", "GET"])
def home():
    if "upload" in request.files: # If the input image is present then we should output a result else render the same home page
        global path, final
        file = request.files["upload"] #Creating a file object from request to save the input image
        img_name = secure_filename(file.filename) # Retrieving the imagename
        path = os.path.join(STORAGEPATH,img_name)
        file.save(path) #Saving the input image into local storage
        image = cv2.imread(path)
        distance = search(image, data) #Searching for similar images based on similar histograms
        distance = sorted([(i,j) for i,j in enumerate(distance)], key = lambda x:x[1]) #Sorting the image distances 
        path=f"/{STORAGEPATH}/{img_name}"
        final = []
        for i,j in distance[:3]: # Top 3 shortest image distances (chi2 - distances)
            final.append(i)
        
        return redirect(url_for("output"))
    else:
       return render_template("home.html")

@app.route("/search", methods = ["GET"])
def output():
    return render_template("output.html", input_path = path, image_paths = imagepaths, idx = final)



if __name__ == "__main__":
    app.run(debug = True)


