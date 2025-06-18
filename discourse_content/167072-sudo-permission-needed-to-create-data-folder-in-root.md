@Vikram Suriyanarayanan said: [@Jivraj](/u/jivraj) [@carlton](/u/carlton) sir please help


When I am downloading the data folder after processing datagen.py , it is trying to download in root folder and it is facing permission error . how can we overcome this ?  

needs sudo permission all the time…  

[![image](https://europe1.discourse-cdn.com/flex013/uploads/iitm/optimized/3X/f/5/f51040627e050b955bb243c23f1f660da36b73ae_2_690x70.png)image2100×216 100 KB](https://europe1.discourse-cdn.com/flex013/uploads/iitm/original/3X/f/5/f51040627e050b955bb243c23f1f660da36b73ae.png "image")


@Carlton D'Silva said: Hi Vikram,


This is because (if you watched the session, or examined the code, you would have realised that) datagen.py was designed to run inside your docker container. And datagen.py (or a similar named file which we will not tell you ahead of time and will be provided as the query parameter in task A1\) will normally be called by evaluate.py  

Inside the docker container, permission for the data folder is set by the Dockerfile  

which then allows your application to access the root folder inside your docker image and create the /data folder.


So the workflow is like this (for your internal testing only… please follow the Project page for deliverables and evaluation to submit project successfully):


1. You create your application server that serves 2 endpoints on localhost:8000
2. You create a docker image that runs this application server.
3. You run the docker image using podman as described in the project page.
4. For mimicking the testing conditions. You need two files:  

evaluate.py and datagen.py to be in the same folder where you are running these two scripts.
5. Run evalute.py using uv.


If your docker image is correctly configured and your application is correctly configured, then all the tasks run by evaluate.py will correctly tell you if the application is producing the right result for each task.


Hope that gives clarity.


Kind regards

