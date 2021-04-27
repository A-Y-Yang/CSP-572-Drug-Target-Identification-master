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

from .models import Similar, Prediction

from IPython.display import Image

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Draw

from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs

from django.contrib.auth.decorators import login_required

import time

import matplotlib
matplotlib.use('Agg')

@login_required(login_url='login')
def index(request):
    template = loader.get_template('drugs/results.html')

    if request.method == 'POST':
        print("Post request is called")

    if request.method == 'GET':
        user_id = request.user.id
        similarities = Similar.objects.filter(user_id=user_id, type='query')

        results = Prediction.objects.filter(user_id=user_id)

        if len(similarities) == 0 or len(results) == 0:
            context = {}
            return HttpResponse(template.render(context, request))
        else:

            similar_compounds = {}
            prob_data = None

            for s in similarities:
                print(s.type, s.image_id, s.smile)
                similar_compounds[s.smile] = []
                response_compounds = Similar.objects.filter(image_id = s.image_id, type = 'response')
                for rc in response_compounds:
                    similar_image = Similar.objects.filter(image_id=rc.image_id,
                                                           smile=rc.smile, type="similar").first()

                    print('RC', rc.smile, rc.type)
                    print('Similar image', similar_image.smile, similar_image.type)
                    new_sim = {'smile': rc.smile, 'similarity': round(rc.similarity, 2),
                               'image': similar_image.name,
                               'query': s.name, 'result': rc.name }
                    similar_compounds[s.smile].append(new_sim)

            targets = []
            for r in results:
                prob_data = r.data
                print(prob_data)
                targets = prob_data[list(prob_data.keys())[0]].keys()

            context = { 'targets': targets, 'prob': prob_data, 'similar': similar_compounds }
            return HttpResponse(template.render(context, request))
