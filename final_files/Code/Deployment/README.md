# dti-web-app

A web app for the work done by Illinois Institute of Technology (IIT) students for drug target identification using CHEMBL database for 
pancreatic cancer targets. To run the application, please follow the following steps:

### 1. Create a rdkit environment:

Since the application is using functions from python's rdkit package, you would be needed
to create a virtual environment for rdkit using the following commands:

(Make sure you have conda installed on your system.)
* conda create -c rdkit -n dti-rdkit-env python=3.7.9 rdkit
* In the above command, the name of environment created is `dti-rdkit-env`

### 2. Clone the repository:

Clone this repository to a local folder of your choice on your local system using git clone https://github.com/kperveen/dti-web-app.git


### 3. Activate rdkit environment and Install all dependencies:

Follow the following commands to activate the rdkit environment and to install all the dependencies.

* conda activate dti-rdkit-env
* pip install -r requirements.txt

### 4. Run the application:

Running the application would require migrations as well. Follow the commands below to run it successfully:

* `python manage.py makemigrations`
* `python manage.py migrate`
* `python manage.py runserver`

The last command will start the server and you can then access the application at (http://127.0.0.1:8000/)

### 5. Create a superuser

You can also create an admin user by running the following command

*  `python manage.py createsuperuser`
* Enter the preferred username and password
* Access the admin interface at (http://127.0.0.1:8000/admin)

