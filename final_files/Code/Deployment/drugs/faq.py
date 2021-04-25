from django.http import HttpResponse
from django.template import loader
from drugs.models import Indication, Therapeutic
import pickle
from padelpy import from_smiles
import pandas as pd
from django.contrib import messages
from django.conf import settings
from django.core.mail import EmailMessage
import dataframe_image as dfi
import tensorflow as tf
from tensorflow import keras
import rdkit
from rdkit import Chem

from IPython.display import Image

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Draw

from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs

from django.contrib.auth.decorators import login_required

import matplotlib
matplotlib.use('Agg')

@login_required(login_url='login')
def index(request):
    if request.method == 'POST':
        print("Get request is called")

    if request.method == 'GET':
        print("Post request is called")

    template = loader.get_template('drugs/faqs.html')

    context = { }
    return HttpResponse(template.render(context, request))
# Create your views here.

