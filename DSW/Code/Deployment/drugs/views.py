from django.http import HttpResponse
from django.template import loader
from django.utils import asyncio
from django.http import HttpResponseRedirect

#from drugs.models import Indications, Therapeutic_areas
import pickle
#from padelpy import from_smiles
import pandas as pd
from django.contrib import messages
from django.conf import settings
from django.core.mail import EmailMessage
import dataframe_image as dfi
from tensorflow import keras
import after_response
import os
import mimetypes
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
import warnings
from django.template.loader import render_to_string, get_template
import time

from .models import Similar, Prediction

import pubchempy as pcp

import numpy as np
import uuid
# from django_ztask.decorators import task

import rdkit
from rdkit import Chem

from IPython.display import Image

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Draw

from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs

import joblib
import matplotlib
matplotlib.use('Agg')



@login_required(login_url='login')
def dashboard(request):
    global flag
    global set_res
    set_res = True
    flag = 0

    drug_dict = dict()


    if request.method == 'GET':
        print('set res is ', set_res)
        user_id = request.user.id
        similarities = Similar.objects.filter(user_id=user_id, type='query')

        results = Prediction.objects.filter(user_id=user_id)

        if 'set_res' in request.session:
            if (len(similarities) > 0 or len(results) > 0) and request.session['set_res'] == True:
                request.session['results'] = True

            elif len(similarities) == 0 or len(results) == 0:
                request.session['results'] = False
        else:
            if len(similarities) > 0 or len(results) > 0:
                request.session['results'] = True

            elif len(similarities) == 0 or len(results) == 0:
                request.session['results'] = False

    if request.method == 'POST':
        request.session['results'] = False
        request.session.modified = True

        flag = 1
        global label, pred_prob
        global email
        email = None
        if request.user.is_authenticated:
            email = request.user.username
        indi = request.POST.get('indication')
        therepu = request.POST.get('therapeutic')
        csv_file = request.FILES['uploadedFile']
        some_var = request.POST.getlist('checks')

        if not csv_file.name.endswith('.csv'):
            messages.error(request, 'File is not csv type')
        else:
            file_data = pd.read_csv(csv_file, index_col=0)
#        df = pd.DataFrame()

        ml_model.after_response(file_data, some_var, email, request, set_res)


        template = loader.get_template('drugs/index.html')
        # context = {'indication_list': indication_list,
        #            'model_list': therapeutic_list,
        #            'flagg': flag,
        #            'emailll': email}
        context = {'flagg': flag,
                   'emailll': email}
        request.session['set_res'] = False
        return redirect('dashboard')


    template = loader.get_template('drugs/index.html')
    context = {'drug_dictionary': drug_dict,
               'flagg': flag}
    return HttpResponse(template.render(context, request))
# Create your views here.



@after_response.enable
def ml_model(file_data, some_var, email, request, set_res):
    warnings.filterwarnings(action="ignore")
    # request.session.calc = True
    print("In ASYNC Func....")
    start = time.time()
    cols = [f'PubchemFP{i}' for i in range(881)]
    df = pd.DataFrame(columns=cols)
    index = 0
    for i in file_data.head(5)['canonical_smiles']:
        c_list = pcp.get_compounds(i, 'smiles')
        fp = c_list[0].cactvs_fingerprint
        fp_arr = np.fromstring(fp, 'u1') - ord('0')
        df.loc[index] = fp_arr
        index += 1
    end = time.time()
    print('elapsed time ', end - start)
    df = df.reset_index()
    df = df.drop('index', axis=1)
    df = df.astype('str').astype('int')
    dictionary = dict()

    # path = "/Users/kausar/Documents/practicum-app/dti_web_app/"
    dict_mlc = dict()
    print(some_var)
    for i in some_var:
        if i == 'EGFR':
            # xpath = os.path.join(path, 'Static/Models/Binary/CHEMBL_203/rf.pkl')
            with open('Static/Models/Binary/CHEMBL_203/CHEMBL203_clf.pkl', 'rb') as file:
                # with open(xpath, 'rb') as file:
                pickle_model = joblib.load(file)
            prob = pickle_model.predict_proba(df)
            dictionary['EGFR'] = list(prob[:, 1]*100)
        elif i == 'IGF1R':
            # xpath = os.path.join(path, 'Static/Models/Binary/CHEMBL_1957/rf.pkl')
            with open('Static/Models/Binary/CHEMBL_1957/CHEMBL1957_clf.pkl', 'rb') as file:
                # with open(xpath, 'rb') as file:
                print("in the section MIA-PaCa")
                pickle_model = joblib.load(file)
            prob = pickle_model.predict_proba(df)
            dictionary['IGF1R'] = list(prob[:, 1]*100)
        elif i == 'MIA-PaCa-2':
            # xpath = os.path.join(path, 'Static/Models/Binary/CHEMBL_614725/rf.pkl')
            with open('Static/Models/Binary/CHEMBL_614725/CHEMBL614725_clf.pkl', 'rb') as file:
                # with open(xpath, 'rb') as file:
                pickle_model = joblib.load(file)
            prob = pickle_model.predict_proba(df)
            dictionary['MIA-PaCa'] = list(prob[:, 1]*100)
        else:
            # xpath = os.path.join(path, 'Static/Models/Binary/CHEMBL_2842/rf.pkl')
            # with open(xpath, 'rb') as file:
            with open('Static/Models/Binary/CHEMBL_2842/CHEMBL2842_clf.pkl', 'rb') as file:
                pickle_model = joblib.load(file)
            prob = pickle_model.predict_proba(df)
            dictionary['mTOR'] = list(prob[:, 1]*100)
    gd = pd.DataFrame(dictionary, index=file_data.head(5)['canonical_smiles'].values)
    # gd = gd.rename_axis(index='Compounds', columns='Targets')
    # xpath = os.path.join(path, 'Static/Models/MLC/mlc_model.h5')
    #from tensorflow.keras.models import load_model 
    import tensorflow as tf
    HeUniform = tf.keras.initializers.he_uniform()
    mlc_model = keras.models.load_model('Static/Models/MLC/mlc_clf.h5', custom_objects={'HeUniform': HeUniform},compile=False)
    prob = mlc_model.predict_proba(df)

    for j in some_var:
        if j == 'EGFR':
            dict_mlc['EGFR'] = list(prob[:, 3]*100)
        elif j == 'IGF1R':
            dict_mlc['IGF1R'] = list(prob[:, 1]*100)
        elif j == 'MIA-PaCa-2':
            dict_mlc['MIA-PaCa'] = list(prob[:, 0]*100)
        elif j == 'mTOR':
            dict_mlc['mTOR'] = list(prob[:, 2]*100)

    gd_mlc = pd.DataFrame(dict_mlc, index=file_data.head(5)['canonical_smiles'].values)

    df_avg = pd.DataFrame(columns=gd.columns)

    if 'EGFR' in gd and 'EGFR' in gd_mlc:
        df_avg['EGFR'] = (gd['EGFR'] + gd_mlc['EGFR']) / 2

    if 'IGF1R' in gd and 'IGF1R' in gd_mlc:
        df_avg['IGF1R'] = (gd['IGF1R'] + gd_mlc['IGF1R']) / 2

    if 'MIA-PaCa' in gd  and 'MIA-PaCa' in gd_mlc:
        df_avg['MIA-PaCa'] = (gd['MIA-PaCa'] + gd_mlc['MIA-PaCa']) / 2

    if 'mTOR' in gd and 'mTOR' in gd_mlc:
        df_avg['mTOR'] = (gd['mTOR'] + gd_mlc['mTOR']) / 2

    joined_df = df_avg
    df_avg = df_avg.rename_axis(index='Compounds', columns='Targets')
    df_styled = df_avg.style.applymap(color_)

    df_styled = df_styled.format({
            'EGFR': '{:,.2f}%'.format,
            'IGF1R': '{:,.2f}%'.format,
            'MIA-PaCa': '{:,.2f}%'.format,
            'mTOR': '{:,.2f}%'.format,
        })
    df_styled = df_styled.set_caption("\nResults")
    dfi.export(df_styled, 'table1.png')
    subject = 'Welcome to "Drogue Cibler"'
    message = 'Hi, {} \n\n Thank you for using Drouge Cibler for drug target ' \
              'identification. \n \n Attached are the results for active/inactive compounds'.format(request.user.first_name)
    email_from = settings.EMAIL_HOST_USER
    recipient_list = [email, ]
    mail = EmailMessage(subject, message, email_from, recipient_list)
    mail.attach_file('table1.png')
    mail.send()
    print("------------------COMPLETED!--------------------")
    print("user id in views.py", request.user.id)
    calculate_similarities(gd, gd_mlc, request, set_res)

def calculate_similarities(gd, gd_mlc, request, sr):
    user_id = request.user.id
    print("user id of current user is ", user_id)
    instances = Similar.objects.filter(user_id=user_id)
    instances.delete()

    results = Prediction.objects.filter(user_id=user_id)
    results.delete()

    gd.reset_index(inplace=True)
    gd = gd.rename(columns={'index': 'Compounds'})

    gd_mlc.reset_index(inplace=True)
    gd_mlc = gd_mlc.rename(columns={'index': 'Compounds'})
    print("calculating similarities", gd.columns, gd_mlc.columns)
    binary_prob = gd
    mlc_prob = gd_mlc
    compounds = binary_prob["Compounds"].tolist()

    prob_data = {}
    targets = binary_prob.columns[1:len(binary_prob.columns)]
    similar_compounds = {}

    target_ids = {
        'EGFR': 'CHEMBL203',
        'IGF1R': 'CHEMBL1957',
        'MIA-PaCa': 'CHEMBL614725',
        'mTOR': 'CHEMBL2842'
    }

    train_data = pd.read_csv('Static/data/train-data.csv')

    k = 0
    start = time.time()
    for c in compounds:
        prob_data[c] = {}
        img_name = uuid.uuid4().hex

        id = uuid.uuid4().hex

        j = 0
        active = False
        for i in range(0, len(targets)):
            prob_data[c][targets[i]] = {}
            prob_model1 = binary_prob[binary_prob["Compounds"] == c][targets[i]].tolist()[0]
            prob_model2 = mlc_prob[mlc_prob["Compounds"] == c][targets[i]].tolist()[0]
            prob_data[c][targets[i]]['probability'] = round(((prob_model1 + prob_model2) / 2), 2)

            if prob_data[c][targets[i]]['probability'] >= 50:
                print("inside if")
                active = True
                tid = target_ids[targets[i]]
                smiles = train_data[train_data["target_chembl_ID"] == tid]["canonical_smiles"].tolist()

                mol1 = Chem.MolFromSmiles(c)
                fp0 = FingerprintMols.FingerprintMol(mol1)
                compound_fig = Draw.MolToMPL(mol1, size=(120, 120))
                # image_cf = img_name + '_qmol_.png'
                # compound_fig.savefig('Static/pictures/similarity/' + image_cf, bbox_inches='tight')
                # #com = Compound()
                # new_image = Similarity(name = image_cf, user_id = request.user.id, type = 'query',
                #                        image_id = id, similarity = 0.0 )
                # new_image.save()
                similar_compounds[c] = []
                for s in smiles:
                    if s != c:
                        mol2 = Chem.MolFromSmiles(s)
                        fp1 = FingerprintMols.FingerprintMol(mol2)

                        similarity = DataStructs.TanimotoSimilarity(fp0, fp1)
                        if similarity > 0.9:
                            fig, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(mol2,
                                                                                           mol1,
                                                                                           SimilarityMaps.GetMorganFingerprint,
                                                                                           metric=DataStructs.TanimotoSimilarity
                                                                                           )
                            name = img_name + '_' + str(j) + '.png'
                            fig.savefig('Static/pictures/similarity/' + name, bbox_inches='tight', pad_inches=0,
                                        transparent=True, orientation="landscape")

                            new_image = Similar(name=name, user_id=user_id, type='similar', image_id=id,
                                                   similarity=similarity, smile=s)
                            new_image.save()

                            r_compound_fig = Draw.MolToMPL(mol2, size=(120, 120))
                            image_rf = img_name + '_rmol_' + str(j) + '.png'
                            r_compound_fig.savefig('Static/pictures/similarity/' + image_rf, bbox_inches='tight')

                            new_image = Similar(name=image_rf, user_id=user_id, type='response', image_id=id,
                                                   similarity=similarity, smile = s)
                            new_image.save()


                            # new_sim = {'smile': s, 'similarity': round(similarity, 2), 'image': name, 'query': image_cf,
                            #            'result': image_rf}
                            # similar_compounds[c].append(new_sim)
                            j += 1
                    matplotlib.pyplot.clf()

        k += 1

        if active == True:
            print("Saving active compound")
            mol1 = Chem.MolFromSmiles(c)
            compound_fig = Draw.MolToMPL(mol1, size=(120, 120))
            image_cf = img_name + '_qmol_.png'
            compound_fig.savefig('Static/pictures/similarity/' + image_cf, bbox_inches='tight')
            # com = Compound()
            new_image = Similar(name=image_cf, user_id=request.user.id, type='query',
                                   image_id=id, similarity=0.0, smile=c)
            new_image.save()

    new_results = Prediction(user_id = request.user.id, data = prob_data)
    new_results.save()
    end = time.time()
    print('elapsed time ', end - start)

    request.session['results'] = True
    request.session.save()

    request.session['set_res'] = True
    request.session.modified = True
    return redirect('dashboard')

    # template = loader.get_template('drugs/index.html')
    # context = {}
    # return HttpResponse(template.render(context, request))

def color_(val):
    if val >= 0.0 and val < 30:
        color = 'red'
    elif val >= 0.3 and val < 80:
        color = 'darkorange'
    else:
        color = 'green'
    return 'color: %s' % color
