# Image-Recogniser-with-Automation
5 well-known celebs are classified using this model built from scratch. This model is then fully automated to have a minimum of 80% accuracy is achieved by deploying different architectures in Docker containers triggered by Jenkins pipeline jobs.

Once the minimum accuracy is achieved, a mail is sent to the Developer with the accuracy and model architecture and the jobs stop running. 
An extra job is also created to monitor so that If container where app is running. fails due to any reason then this job should automatically start the container again from where the last trained model left.

Find the Jupyter notebook and model.py file containing the model trained.
