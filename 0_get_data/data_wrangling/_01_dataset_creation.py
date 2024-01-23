import pandas as pd
import numpy as np
import category_encoders as ce
import re
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import dask.dataframe as dd
from data_wrangling.auxiliars import *
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
import datetime
import os


def preprocess_andalucia():

    print("Step 1: Data ingestion/load")

    paths_dic = {"socio_demo": "0_get_data/data/01_Datos_sociodemo.txt",
                 "ingreso_hosp": "0_get_data/data/03_Ingreso_hosp.txt",
                 "pmhx": "0_get_data/data/02_Patologias_cronicas.txt",
                 "lab": "0_get_data/data/05_MPA.txt",
                 "periodos": "0_get_data/data/05_Periodos_pandemicos.csv",
                 "vacunas": "0_get_data/data/04_Vacunacion_covid.txt"}

    if not exist_pkl("0_get_data/data"):
        socio_demo, ingreso_hosp, pmhx, lab, periodos, vacunas = data_ingestion(
            paths_dic)
    else:
        socio_demo, ingreso_hosp, pmhx, lab, periodos, vacunas = load_data(
            paths_dic)

    print("Step 2: Map hospitals to provincias")
    ingreso_hosp = get_provincia(ingreso_hosp)

    print("Step 3: Merge sociodemo & ingreso -> patient")
    patient = pd.merge(socio_demo, ingreso_hosp,
                       on='NUHSA_ENCRIPTADO', how='inner')
    
    patient = fix_dates(patient)
    print(f"Total patients after sociodemo & ingreso merge: {len(patient)}")

    # Keep only the last admission per patient. If the time difference between admissions (if there's more than one)
    # is 50 or greater, consider the patient as reinfected (1).
    print("Step 4: Manage cases with multiple admissions")
    patient = get_reinfected(patient)
    print(
        f"Total patients rows after managing many admissions: {len(patient)}\n")

    print("Step 5: Remove from patient NUHSA that do not have pmhx or lab") 
    patient = remove_nuhsa_without_lab(patient, pmhx, lab) # Edited only remove patients without conmorbilities
    print(f"Total patients without pmhx entry: {len(patient)}\n")

    print("Step 6: Get age")
    patient['age'] = calculate_patient_age(patient)

    ##### LAB VARIABLES NOT USED #####
    #print("Step 7: Get lab")
    #lab_24h = get_lab(lab, patient)
    print("Step 8: Get pmhx")
    pmhx_encoded = get_pmhx(pmhx, patient)

    #print("Step 9: Merge lab")
    #patient = patient.merge(
    #    lab_24h, on=['NUHSA_ENCRIPTADO', 'admission_datetime'], how='inner')
    #print(f"Total patients after merging lab: {len(patient)}")

    #percentile = 0.95
    #lab_cols = patient.filter(like='lab_')
    #for column in lab_cols.columns:
    #    lower_limit = np.percentile(
    #        patient[column], 100 * (1 - percentile) / 2)
    #    upper_limit = np.percentile(
    #        patient[column], 100 * (1 + percentile) / 2)
    #    patient[column] = np.clip(patient[column], lower_limit, upper_limit)

    print("Step 10: Merge pmhx")
    patient = patient.merge(pmhx_encoded, on='NUHSA_ENCRIPTADO', how='inner')
    print(f"Total patients after merging pmhx: {len(patient)}")

    patient = fill_null_pmhx(patient)

    perform_sanity_check(patient)

    print("Step 11: Get periodo")
    patient = get_periodo(patient, periodos)

    patient = create_hospital_outcome(patient)
    patient = calculate_inpatient_days(patient)
    patient = change_data_types(patient)
    patient = exclusion_criteria(patient, periodos)
    centro_patient_count = patient.groupby('centro').size()
    print(centro_patient_count)

    print("Step 13: Get vacunas")
    patient = get_vacunas(patient, vacunas)

    print("Step 14: Process variables")
    # Create variable delta_days_death
    patient = create_delta_death(patient)

    patient = remove_unused_cols(patient)
    check_nulls(patient)
    print("The 'COD_FEC_FALLECIMIENTO' and 'delta_days_death' columns have more than 50% missing values, indicating that some patients did not pass away.")

    print("Step 15: add type center")
    patient = add_type_center(patient)

    print("Step 16: Save the files")
    
    # Save the files processed
    out_folder = "0_get_data/data/out"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    patient.to_csv(out_folder + "/patient.csv")
    patient.to_pickle(out_folder + "/patient.pkl")
    patient = impute(patient)
    patient.to_csv(out_folder+"/patient_imputed.csv")
    patient.to_pickle(out_folder+"/patient_imputed.pkl")

    print("FINISH SUCCESSFULLY")

# Custom date parser function
def parse_date(date_str):
    date = pd.to_datetime(date_str, format="%Y%m%d")
    return date.strftime("%Y-%m-%d") if not pd.isna(date) else ''

def exist_pkl(path):
    files = set(os.listdir(path))

    pkl_files = set(["socio_demo.pkl",
                    "ingreso_hosp.pkl",
                     "pmhx.pkl",
                     "lab.pkl",
                     "vacunas.pkl"])

    return pkl_files.issubset(files)


def data_ingestion(paths_dict):

    socio_demo = dd.read_csv(paths_dict.get("socio_demo"),
                             sep='|', parse_dates=['COD_FEC_NACIMIENTO'],
                             date_parser=parse_date)
    socio_demo = socio_demo.compute()
    socio_demo = socio_demo.rename(columns={'COD_SEXO': 'sex'})
    socio_demo.to_pickle("0_get_data/data/socio_demo.pkl")
    print("Finished sociodemo")

    ingreso_hosp = dd.read_csv(paths_dict.get("ingreso_hosp"),
                               sep='|',
                               parse_dates=['COD_FEC_ALTA', 'COD_FEC_INGRESO'],
                               date_parser=parse_date)
    ingreso_hosp = ingreso_hosp.compute()
    ingreso_hosp.rename(columns={'DESC_HOSPITAL_NORMALIZADO': 'centro',
                                 'COD_FEC_INGRESO': 'admission_datetime',
                                 'COD_FEC_ALTA': 'discharge_datetime'},
                        inplace=True)
    print("Finished ingreso_hosp")
    ingreso_hosp.to_pickle("0_get_data/data/ingreso_hosp.pkl")

    pmhx = dd.read_csv(paths_dict.get("pmhx"),
                       sep='|', parse_dates=['COD_FEC_INI_PATOLOGIA'],
                       date_parser=parse_date)
    pmhx = pmhx.compute()
    pmhx = pmhx.drop_duplicates()
    pmhx.to_pickle("0_get_data/data/pmhx.pkl")
    print("Finished pmhx")

    lab = dd.read_csv(paths_dict.get("lab"),
                      sep='|',
                      parse_dates=['FEC_EXTRAE'],
                      date_parser=parse_date)
    lab = lab.compute()
    lab = lab.drop_duplicates()
    lab.to_pickle("0_get_data/data/lab.pkl")
    print("Finish lab")

    vacunas = dd.read_csv(paths_dict.get("vacunas"),
                          sep='|', parse_dates=['FEC_VACUNACION'],
                          date_parser=parse_date)
    vacunas = vacunas.compute()
    vacunas = vacunas.drop_duplicates()
    vacunas.to_pickle("0_get_data/data/vacunas.pkl")
    print("Finished vacunas")

    periodos = pd.read_csv(paths_dict.get("periodos"), sep=';')
    periodos['fecha_inicio'] = pd.to_datetime(periodos['fecha_inicio'])
    periodos['fecha_fin'] = pd.to_datetime(periodos['fecha_fin'])

    return socio_demo, ingreso_hosp, pmhx, lab, periodos, vacunas


def load_data(paths_dict):
    socio_demo = pd.read_pickle("0_get_data/data/socio_demo.pkl")
    ingreso_hosp = pd.read_pickle("0_get_data/data/ingreso_hosp.pkl")
    ingreso_hosp.rename(
        columns={'IND_INGRESO_UCI': 'uci', 'NUM_DIAS_UCI': 'u_stay'}, inplace=True)
    pmhx = pd.read_pickle("0_get_data/data/pmhx.pkl")
    lab = pd.read_pickle("0_get_data/data/lab.pkl")
    vacunas = pd.read_pickle("0_get_data/data/vacunas.pkl")

    periodos = pd.read_csv(paths_dict.get("periodos"), sep=';')
    periodos['fecha_inicio'] = pd.to_datetime(periodos['fecha_inicio'])
    periodos['fecha_fin'] = pd.to_datetime(periodos['fecha_fin'])

    return socio_demo, ingreso_hosp, pmhx, lab, periodos, vacunas


def fix_dates(patient):
    patient['admission_datetime'] = pd.to_datetime(
        patient['admission_datetime'].dt.date)
    patient['discharge_datetime'] = pd.to_datetime(
        patient['discharge_datetime'].dt.date)
    patient['COD_FEC_NACIMIENTO'] = pd.to_datetime(
        patient['COD_FEC_NACIMIENTO'].dt.date)
    # COD_FEC_FALLECIMIENTO with -1 is encoded as nan, keeping only the date and not the time
    patient['COD_FEC_FALLECIMIENTO'] = patient['COD_FEC_FALLECIMIENTO'].apply(
        lambda x: parse_date(x) if x != -1 else np.nan)
    patient['COD_FEC_FALLECIMIENTO'] = pd.to_datetime(
        patient['COD_FEC_FALLECIMIENTO'], errors='coerce')
    patient['COD_FEC_FALLECIMIENTO'] = pd.to_datetime(
        patient['COD_FEC_FALLECIMIENTO'].dt.date)
    print(patient.info())
    return patient


def manage_edge_cases(patient):
    # edge cases happen in multiple admissions
    multiple_admissions = patient[patient['NUHSA_ENCRIPTADO'].duplicated(
        keep=False)]
    multiple_admissions['admission_count'] = multiple_admissions.groupby(
        'NUHSA_ENCRIPTADO')['NUHSA_ENCRIPTADO'].transform('count')
    multiple_admissions = multiple_admissions.sort_values(
        'NUHSA_ENCRIPTADO', ascending=False)
    days_between = (multiple_admissions['COD_FEC_FALLECIMIENTO'] -
                    multiple_admissions['discharge_datetime']).dt.days
    multiple_admissions.insert(9, 'days_between', days_between, True)

    grouped = multiple_admissions.groupby('NUHSA_ENCRIPTADO')

    print(
        f"Start dropping edge cases. Multiple admissions initially: {len(multiple_admissions)}")

    multiple_admissions[multiple_admissions['admission_count'] > 2].to_excel(
        './inspect_more.xlsx')
    multiple_admissions = multiple_admissions[multiple_admissions['admission_count'] == 2]
    print(f"After dropping admission_count > 2: {len(multiple_admissions)}")

    corrected_patient = pd.DataFrame(columns=patient.columns)
    multiple_admissions['case'] = None

    for patient_id, patient_group in grouped:
        hospital_1 = patient_group['centro'].iloc[0]
        hospital_2 = patient_group['centro'].iloc[1]
        admission_2 = patient_group['admission_datetime'].max()
        admission_1 = patient_group['admission_datetime'].min()
        discharge_1 = patient_group['discharge_datetime'].min()
        discharge_2 = patient_group['discharge_datetime'].max()
        expire_date = patient_group['COD_FEC_FALLECIMIENTO'].min()
        expire_date = expire_date if isinstance(
            expire_date, datetime.date) and not pd.isna(expire_date) else None
        assign_case = multiple_admissions[multiple_admissions['NUHSA_ENCRIPTADO']
                                          == patient_id].index
        case = 'none'
        if admission_2 == discharge_1:
            if (hospital_1 != hospital_2):
                # case 1: tralsado de hospital
                case = '1'
                patient_group.drop(patient_group.index, inplace=True)
        elif (admission_2 - admission_1).days in [0, 1]:
            if (hospital_1 == hospital_2):
                # case 2_a: vuelve al mismo hospital a los 24h del alta, o el mismo día, se considera un solo ingreso
                case = '2_a'
                patient_group.loc[patient_group['admission_datetime']
                                  == admission_2, 'admission_datetime'] = admission_1
                patient_group.drop(
                    patient_group[patient_group['admission_datetime'] == admission_1].index, inplace=True)
            else:
                # case 2_b: reingresa en los siguientes 24h pero en otro hospital... caso no contemplado
                case = '2_b'
                patient_group.drop(patient_group.index, inplace=True)
        elif expire_date:
            days_between_1 = patient_group.loc[patient_group['admission_datetime']
                                               == admission_1, 'days_between'].iloc[0]
            days_between_2 = patient_group.loc[patient_group['admission_datetime']
                                               == admission_2, 'days_between'].iloc[0]
            if days_between_1 > 30:
                # case X: fallece y tiene dos ingresos diferentes
                # le quitamos la fecha de fallecimiento al primer ingreso para que le contabilice al segundo (???)
                case = 'X'
                patient_group.loc[patient_group['admission_datetime']
                                  == admission_1, 'COD_FEC_FALLECIMIENTO'] = ''
        elif (admission_2 - admission_1).days == 0 and (discharge_1 - discharge_2).days != 0:
            if (hospital_1 != hospital_2):
                # case 4_b: si es de hospitales distintos y aparece como que tiene dos ingresos en un mismo día
                # me quedo con el que dura más
                case = '4_b'
                patient_group.drop(
                    patient_group[patient_group['discharge_datetime'] == discharge_1].index, inplace=True)
        elif (admission_2 - admission_1).days == 0 and (discharge_1 - discharge_2).days == 0:
            # case5: si la fecha de alta y baja de los dos ingresos es la misma, descartamos...
            case = '5'
            patient_group.drop(patient_group.index, inplace=True)
        else:
            case = 'none'

        multiple_admissions.loc[assign_case, 'case'] = case
        corrected_patient = corrected_patient.append(patient_group)

    corrected_patient = corrected_patient.append(
        patient[patient['NUHSA_ENCRIPTADO'].duplicated(keep=False) == False])
    corrected_patient = corrected_patient.sort_values(
        ['NUHSA_ENCRIPTADO', 'admission_datetime'])
    corrected_patient = corrected_patient.reset_index(drop=True)

    return corrected_patient

# Define a custom function to calculate days between


def calculate_days_between(row):
    if pd.notnull(row['COD_FEC_FALLECIMIENTO']) and pd.notnull(row['discharge_datetime']):
        return (row['COD_FEC_FALLECIMIENTO'] - row['discharge_datetime']).days
    else:
        return None


def create_hospital_outcome(patient):
    # patients with fecha_fallecimiento need to be divided into:  1) hospital_outcome = 1 if expires at 30 days from discharge
    # 2) hospital_outcome = 1 if expires one day before discharge
    # 3) hospital_outcome = 0 else

    # Apply the custom function to create the 'days_between' column
    patient['COD_FEC_FALLECIMIENTO'] = pd.to_datetime(
        patient['COD_FEC_FALLECIMIENTO'])
    patient['discharge_datetime'] = pd.to_datetime(
        patient['discharge_datetime'])
    patient['aux'] = (patient['COD_FEC_FALLECIMIENTO'] -
                      patient['discharge_datetime']).dt.days

    hospital_outcome = np.where(
        (patient['aux'] >= -365) & (patient['aux'] <= 30), 1, 0)
    patient.insert(6, 'hospital_outcome', hospital_outcome, True)

    delete_faulty_rows = patient['aux'] < -365
    patient = patient[~delete_faulty_rows]

    patient.reset_index(drop=True, inplace=True)

    return patient.drop(columns=['aux'])


def calculate_patient_age(patient):
    return [relativedelta(a, b).years for a, b in zip(patient['admission_datetime'], patient['COD_FEC_NACIMIENTO'])]


def fill_null_pmhx(patient):
    fill_cols = patient.filter(like='pmhx').columns
    patient[fill_cols] = patient[fill_cols].fillna(value=0)
    return patient


def study_uci_stay(patient):
    grouped = patient.groupby('centro')
    result = grouped['u_stay'].apply(
        lambda x: round((x.isnull().sum() / len(x)) * 100, 1))
    result.to_latex("../tables_figures/task_tables/uci_days_nulls.tex")


def study_multiple_stay(patient, ids_to_remove):
    total_counts = patient.groupby('centro')['NUHSA_ENCRIPTADO'].nunique()
    ids_df = patient[patient['NUHSA_ENCRIPTADO'].isin(ids_to_remove)]
    centro_counts = ids_df.groupby('centro')['NUHSA_ENCRIPTADO'].nunique()
    centro_percentage = (centro_counts / total_counts) * 100
    centro_counts_df = centro_counts.reset_index(name='count_ids')
    centro_counts_df['percentage'] = centro_percentage.reset_index(drop=True)
    centro_counts_df.drop(columns=['count_ids'], inplace=True)
    centro_counts_df.to_latex(
        "../tables_figures/task_tables/multiple_stays.tex")


def check_nulls(patient):
    threshold = len(patient) * 0.5  # 50% threshold
    columns_with_nulls = patient.columns[patient.isnull().sum() > threshold]
    print(f'Columns with more than 0.5 null {columns_with_nulls}')


def get_lab(lab, patient):
    # Convert non-numeric values to NaN
    lab["VALOR_CONVENCIONAL"] = pd.to_numeric(
        lab["VALOR_CONVENCIONAL"].str.replace(",", ".", regex=True), errors='coerce')
    # remove whitespaces from the beginning and end of the strings
    lab['DESC_CLC'] = lab['DESC_CLC'].str.strip()
    new_lab_names = list(set(list(lab_map.values())))
    lab['DESC_CLC'] = lab['DESC_CLC'].replace(lab_map)

    lab_24h = pd.merge(patient[['NUHSA_ENCRIPTADO', 'admission_datetime']],
                       lab, on='NUHSA_ENCRIPTADO', how='inner')
    condition = lab_24h['FEC_EXTRAE'] <= (
        lab_24h['admission_datetime'] + pd.Timedelta(days=2))
    lab_24h = lab_24h[condition]

    lab_24h = lab_24h[lab_24h['DESC_CLC'].isin(new_lab_names)]
    lab_24h['VALOR_CONVENCIONAL'] = np.where(
        lab_24h['VALOR_CONVENCIONAL'] < 0, np.nan, lab_24h['VALOR_CONVENCIONAL'])

    # Define a custom filtering function
    def filter_rows(group):
        exact_match = group[group['admission_datetime'] == group['FEC_EXTRAE']]
        if not exact_match.empty:
            group['date_match'] = 1  # Set value to 1 for exact match
            return exact_match
        closest_match_index = (
            group['FEC_EXTRAE'] - group['admission_datetime']).abs().idxmin()
        return group.loc[[closest_match_index]]

    filtered_df = lab_24h.groupby(
        ['NUHSA_ENCRIPTADO', 'admission_datetime'], group_keys=False, as_index=False).apply(filter_rows)

    pivot_table_df = filtered_df.pivot_table(values='VALOR_CONVENCIONAL',
                                             index=['NUHSA_ENCRIPTADO',
                                                    'admission_datetime'],
                                             columns='DESC_CLC',
                                             aggfunc='first')
    pivot_table_df = pivot_table_df.reset_index()
    return pivot_table_df


def get_pmhx(pmhx, patient):
    pmhx['COD_FEC_FIN_PATOLOGIA'] = pmhx['COD_FEC_FIN_PATOLOGIA'].apply(
        lambda x: parse_date(x) if x != -1 else np.nan)
    pmhx['DESC_PATOLOGIA_CRONICA'] = pmhx['DESC_PATOLOGIA_CRONICA'].map(
        lambda x: x if x in [value for values in dx_dict.values() for value in values] else '')
    pmhx['DESC_PATOLOGIA_CRONICA'] = pmhx['DESC_PATOLOGIA_CRONICA'].map(
        {value: key for key, values in dx_dict.items() for value in values})

    ce_OHE = ce.OneHotEncoder(use_cat_names=True, cols=[
                              'DESC_PATOLOGIA_CRONICA'])
    pmhx_encoded = ce_OHE.fit_transform(pmhx)
    pmhx_encoded.rename(columns=lambda x: re.sub(
        r'.*?(pmhx.*)', r'\1', x) if 'pmhx' in x else x, inplace=True)

    patient_pmhx = patient[['COD_FEC_FALLECIMIENTO',
                            'admission_datetime',
                            'NUHSA_ENCRIPTADO']].merge(pmhx_encoded, on='NUHSA_ENCRIPTADO', how='outer')
    # outer if we assume patients that do not appear in pmhx have no comorbilities (need to fill nan with 0's)
    # inner if we think the information is missing

    patient_pmhx.drop(columns=['COD_FEC_FALLECIMIENTO',
                               'admission_datetime',
                               'COD_PATOLOGIA_BPS',
                               'DESC_PATOLOGIA_CRONICA_nan',
                               'COD_FEC_INI_PATOLOGIA',
                               'COD_FEC_FIN_PATOLOGIA'],
                        inplace=True)
    patient_pmhx = patient_pmhx.groupby('NUHSA_ENCRIPTADO').max().reset_index()
    

    return patient_pmhx


def get_periodo(patient, periodos):
    conditions = []
    for i, row in periodos.iterrows():
        mask = (patient['admission_datetime'] >= row['fecha_inicio']) & (
            patient['admission_datetime'] <= row['fecha_fin'])
        conditions.append(mask)

    periodo_names = periodos['nombre_periodo'].apply(lambda x: x.split(" ")[1])
    patient['periodo'] = np.select(
        conditions, periodo_names, default='Unknown')

    # Use one-hot encoding to transform the 'periodo' column
    ce_OHE = ce.OneHotEncoder(use_cat_names=True, cols=['periodo'])
    patient = ce_OHE.fit_transform(patient)

    return patient


def merge_lab_pmhx(pmhx_encoded, lab_24h):
    nuhsa_not_in_lab = pmhx_encoded[~pmhx_encoded['NUHSA_ENCRIPTADO'].isin(
        lab_24h['NUHSA_ENCRIPTADO'])]['NUHSA_ENCRIPTADO'].unique()  # nuhsa in patient and not in pmhx
    nuhsa_not_in_pmhx = lab_24h[~lab_24h['NUHSA_ENCRIPTADO'].isin(
        pmhx_encoded['NUHSA_ENCRIPTADO'])]['NUHSA_ENCRIPTADO'].unique()  # nuhsa in patient and not in lab

    print("NUHSA patient but not in pmhx: ", len(nuhsa_not_in_pmhx))
    print("NUHSA patient but not in lab: ", len(nuhsa_not_in_lab))
    lab_pmhx = pmhx_encoded.merge(lab_24h, on='NUHSA_ENCRIPTADO', how='right')
    lab_pmhx.drop(columns=['admission_datetime',
                  'COD_FEC_FALLECIMIENTO'], inplace=True)
    return lab_pmhx


def get_vacunas(patient, vacunas):
    patient.insert(9, 'num_shots', 0)
    for i in vacunas.index:
        nuhsa = vacunas['NUHSA_ENCRIPTADO'][i]
        rows = patient.loc[patient['NUHSA_ENCRIPTADO'] == nuhsa]
        # leer la fecha de ingreso y ver cuantas vacunas tiene puestas, actualizar num_shots
        shot_date = vacunas['FEC_VACUNACION'][i]
        shot_date_plus_14d = shot_date + timedelta(days=14)
        for _, row in rows.iterrows():
            admission_datetime = row['admission_datetime']
            if shot_date_plus_14d < admission_datetime:
                patient.loc[(patient['NUHSA_ENCRIPTADO'] == nuhsa) & (
                    patient['admission_datetime'] == admission_datetime), 'num_shots'] += 1

    return patient


def get_provincia(ingreso_hosp):
    # remove whitespaces from the beginning and end of the strings
    ingreso_hosp['centro'] = ingreso_hosp['centro'].str.strip()
    ingreso_hosp['provincia'] = ingreso_hosp['centro'].replace(
        hospital_province_map)
    # ingreso_hosp['comunidad'] = 'Andalucía'
    ingreso_hosp.drop(columns=['COD_HOSPITAL_NORMALIZADO'], inplace=True)
    return ingreso_hosp


def remove_unused_cols(patient):
    cols_to_remove = ['COD_FEC_NACIMIENTO',
                      #'NUHSA_ENCRIPTADO',# CAN BE USEFUL
                      #'discharge_datetime',# CAN BE USEFUL
                      'DESC_SEXO', #0:man, 1:woman
                      #'COD_FEC_FALLECIMIENTO', # CAN BE USEFUL
                      'pmhx_tobacco', #unreliable
                      'u_stay'# Many nulls
                      ]
    return patient.drop(columns=cols_to_remove)


def exclusion_criteria(patient, periodos):
    periodos_start = periodos["fecha_inicio"].iloc[0]
    print(periodos_start)
    periodos_finish = periodos["fecha_fin"].iloc[-1]
    print(periodos_finish)

    print('\n:::Exclusion criteria:::')
    counts = patient['NUHSA_ENCRIPTADO'].value_counts()
    print(f'Initial number of cases: {len(patient)}')
    patient = patient[patient['age'] >= 18]
    print(f'After excluding those with <18 Age: {len(patient)}')
    # Not drop patients  from any period 
    #patient = patient[patient['periodo_1'] == 0]
    #patient = patient.drop(columns=['periodo_1'])
    #print(f'After excluding those from periodo 1: {len(patient)}')


    # Not drop patients if there are less than 150 cases in the center
    # patient = patient.groupby('centro').filter(lambda x: len(x) >= 150)
    # print(
    #     f'After excluding hospitals with less than 150 patients: {len(patient)}')


    # just remove patients with less than 0 days in hospital
    patient = patient[#(patient['inpatient_days'] <= 30) &
                      (patient['inpatient_days'] >= 0)]

    print(
        #f'After excluding hospitals with more than 30 days inpatient or less than 0: {len(patient)}')
        f'After excluding hospitals with less than 0: {len(patient)}')

    patient = patient[patient["admission_datetime"].isnull() != True]
    print(f'After excluding those with missing admission time: {len(patient)}')

    patient = patient[patient['hospital_outcome'].isnull() == False]
    print(
        f'After excluding those with missing hospital_outcome: {len(patient)}')

    print(f'Hospitals left: {patient["centro"].nunique()}')
    # Outcome distribution
    print(':::Outcome distribution:::')
    print('Breakdown of hospital_outcome:')
    outcome_counts = patient['hospital_outcome'].value_counts()
    print(outcome_counts)
    return patient


def change_data_types(patient):
    for col in patient.columns:
        if col.startswith('lab_') or col == 'u_stay':
            patient.loc[patient[col] == ".", col] = np.nan
            patient.loc[patient[col] == "", col] = np.nan
            patient[col] = patient[col].astype('float64')
        elif any(keyword in col for keyword in ['pmhx', 'sex', 'age', 'periodo', 'uci', 'provincia_', 'prior_admission_14', 'num_shots', 'prior_admission', 'inpatient_days']):
            patient[col] = patient[col].astype(int)

    return patient


def impute(patient):
    print(":::Performing imputation with IterativeImputer:::")
    exclude_columns = ["COD_FEC_FALLECIMIENTO", "delta_days_death"] # Nulls encoded no death

    any_missing = patient.columns[patient.isnull().sum() > 0].tolist()
    any_missing = list(set(any_missing) - set(exclude_columns))
    print(any_missing)
    print('Number of nulls before imputation: ', patient.drop(exclude_columns, axis=1).isnull().values.sum())

    impute_pipeline = Pipeline(steps=[
        ('impute', IterativeImputer(random_state=42, min_value=0)),
    ])

    col_trans = ColumnTransformer(transformers=[
        ('impute_pipeline', impute_pipeline, any_missing)],
        remainder='passthrough',
        n_jobs=-1,
        verbose_feature_names_out=False)

    col_trans.set_output(transform='pandas')
    patient = col_trans.fit_transform(patient)
    print(
        f'Number of nulls after imputation: {patient.drop(exclude_columns, axis=1).isnull().values.sum()}\n')
    patient = patient.round(2)
    return patient


def calculate_inpatient_days(patient):
    print(patient.info())
    patient['inpatient_days'] = np.where(
        patient['hospital_outcome'] == 0,
        (patient['discharge_datetime'] -
         patient['admission_datetime']).dt.days,
        np.where(
            (patient['hospital_outcome'] == 1) & (
                patient['COD_FEC_FALLECIMIENTO'] == patient['discharge_datetime']),
            (patient['discharge_datetime'] -
             patient['admission_datetime']).dt.days,
            (patient['COD_FEC_FALLECIMIENTO'] -
             patient['admission_datetime']).dt.days
        )
    )
    return patient


def is_number(value):
    try:
        return value.replace(',', '.').isdigit() or value.replace(',', '.').isnumeric() or value.replace(',', '.').isdecimal()
    except Exception as e:
        return f"Error: {e}"


def remove_nuhsa_without_lab(patient, pmhx, lab):
    nuhsa_not_in_pmhx = patient[~patient['NUHSA_ENCRIPTADO'].isin(
        pmhx['NUHSA_ENCRIPTADO'])][['centro', 'NUHSA_ENCRIPTADO']]
    ##### NOT USED LAB #####
    #nuhsa_not_in_lab = patient[~patient['NUHSA_ENCRIPTADO'].isin(
    #    lab['NUHSA_ENCRIPTADO'])][['centro', 'NUHSA_ENCRIPTADO']]

    print("NUHSA patient but not in pmhx:", len(nuhsa_not_in_pmhx))
    #print("NUHSA patient but not in lab:", len(nuhsa_not_in_lab))

    def calculate_null_percentage(grouped_df):
        total_records = len(grouped_df)
        #nulls_lab = nuhsa_not_in_lab[nuhsa_not_in_lab['centro']
        #                             == grouped_df['centro'].iloc[0]].shape[0]
        nulls_pmhx = nuhsa_not_in_pmhx[nuhsa_not_in_pmhx['centro']
                                       == grouped_df['centro'].iloc[0]].shape[0]
        #percentage_nulls_lab = (nulls_lab / total_records) * 100
        percentage_nulls_pmhx = (nulls_pmhx / total_records) * 100
        return pd.Series({
            # Include the 'province' column
            'Provincia': grouped_df['provincia'].iloc[0],
            'Número Ingresos': int(total_records),
        #    'Ausentes Laboratorio (%)': percentage_nulls_lab,
            'Ausentes Comorbilidades (%)': percentage_nulls_pmhx
        })

    # Group by 'centro' and apply the function
    result_df = patient.groupby('centro', group_keys=False).apply(
        calculate_null_percentage)
    result_df['Número Ingresos'] = result_df['Número Ingresos'].apply(
        lambda x: format(x, ',.0f').replace(',', '.'))
    result_df = result_df.sort_values(by='Provincia')

    #result_df.to_latex(
    #    'E:/Phd_DataResearch_COVID/rascal_migrate/tables_figures/missings/missing_lab_pmhx.tex')

    mask = result_df['Ausentes Comorbilidades (%)'] > 50
    a = result_df[mask]
    a = a.reset_index()
    centros_to_remove = a['centro'].tolist()
    patient = patient[~patient['centro'].isin(centros_to_remove)]

    return patient


def get_reinfected(patient, threshold_days_reinfection=50):

    # Sorting by 'NUHSA_ENCRIPTADO' and 'admission_datetime'
    patient.sort_values(
        ["NUHSA_ENCRIPTADO", "admission_datetime"], inplace=True)

    # Calculating the maximum difference in days for each patient
    patient["max_diff_admission"] = patient.groupby("NUHSA_ENCRIPTADO")[
        "admission_datetime"].transform(lambda x: (x.max() - x.min()).days)

    # Labeling patients as 'reinfected' based on the threshold
    patient["reinfected"] = (patient["max_diff_admission"]
                             >= threshold_days_reinfection).astype(int)

    # Converting 'reinfected' to a categorical (factor) column
    patient["reinfected"] = pd.Categorical(
        patient["reinfected"], categories=[0, 1])

    # Selecting the last entry for each patient
    # Patients with 'reinfected' value of 1 are considered reinfected
    patient = patient.groupby("NUHSA_ENCRIPTADO").last().reset_index()

    # Dropping the unnecessary column
    patient.drop(columns=["max_diff_admission"], inplace=True)

    # Show the results
    n_reinfected = (patient["reinfected"] == 1).sum()
    n_no_reinfected = (patient["reinfected"] == 0).sum()
    print(
        f"\nDays difference in admission date to consider reinfection: {threshold_days_reinfection}")
    print(
        f"Reinfected -> {(n_reinfected/(n_reinfected+n_no_reinfected)*100):.3f}% ({n_reinfected})")
    print(
        f"No reinfected -> {(n_no_reinfected/(n_reinfected+n_no_reinfected)*100):.3f}% ({n_no_reinfected})\n")

    return patient


def perform_sanity_check(patient):
    print("\nSanity check")
    print(f'First admission: {patient["admission_datetime"].min()}')
    print(f'Last admission: {patient["admission_datetime"].max()}')
    print(f'First discharge: {patient["discharge_datetime"].min()}')
    print(f'Last discharge: {patient["discharge_datetime"].max()}')
    print(
        f'Missing admission time: {patient["admission_datetime"].isnull().sum()}')
    print(
        f'Missing discharge time: {patient["discharge_datetime"].isnull().sum()}')

def create_delta_death(patient):
    
    patient["delta_days_death"] = (patient["COD_FEC_FALLECIMIENTO"] - patient["admission_datetime"]).dt.days
    # If not dead, null in the variable
    patient["delta_days_death"].where(patient["COD_FEC_FALLECIMIENTO"].notna(), inplace=True)

    return patient

def add_type_center(patients):
    types_df = pd.read_excel("0_get_data/data/06_hospital_types.xlsx")
    types_df = types_df[['Hospital', "Categoría"]]

    hospital_mapper = {'Hospital Universitario Torrecárdenas': 'H.U. Torrecárdenas',
                'Hospital de Poniente-El Ejido': 'H. de Poniente-El Ejido',
                'Hospital La Inmaculada': 'H. La Inmaculada',
                'Hospital El Toyo': 'H. El Toyo',
                'Hospital Universitario Puerta del Mar': 'H.U. Puerta del Mar',
                'Hospital de Puerto Real': 'H.U. de Puerto Real',
                'Hospital de Jerez de la Frontera': 'H.U. de Jerez de la Frontera',
                'Hospital Punta de Europa': 'H. Punta de Europa',
                'Hospital de La Línea de la Concepción': 'H. de La Línea de la Concepción',
                'Hospital de Alta Resolución La Janda': 'h. de Alta Resolución La Janda',
                'Hospital Universitario Reina Sofía': 'H.U. Reina Sofía',
                'Hospital Valle de los Pedroches': 'H. Valle de los Pedroches',
                'Hospital Infanta Margarita': 'H. Infanta Margarita',
                'Hospital de Puente Genil': 'H. de Puente Genil',
                'Hospital Valle del Guadiato': 'H. Valle del Guadiato',
                'Hospital Universitario Virgen de las Nieves': 'H.U. Virgen de las Nieves',
                'Hospital Universitario San Cecilio': 'H.U. San Cecilio',
                'Hospital de Baza': 'H. de Baza',
                'Hospital Santa Ana': 'H. Santa Ana',
                'Hospital de Guadix': 'H. de Guadix',
                'Hospital de Loja': 'H. de Loja',
                'Hospital Universitario Juan Ramón Jiménez': 'H.U. Juan Ramón Jiménez',
                'Hospital Infanta Elena': 'H. Infanta Elena',
                'Hospital de Riotinto': 'H. de Riotinto',
                'Hospital de la Costa Occidental de Huelva': 'H. de la Costa Occidental de Huelva',
                'Hospital del Condado': 'H. del Condado',
                'Hospital de la Sierra': 'H. de la Sierra',
                'Complejo Hospitalario de Jaén': 'H.U. de Jaén',
                'Hospital Alto Guadalquivir de Andújar': 'H. Alto Guadalquivir de Andújar',
                'Hospital San Agustín': 'H. San Agustín',
                'Hospital San Juan de la Cruz': 'H. San Juan de la Cruz',
                'Hospital de Alcaudete': 'H. de Alcaudete',
                'Hospital Sierra de Segura': 'H. Sierra de Segura',
                'Hospital de Alcalá la Real': 'H. de Alcalá la Real',
                'Hospital Regional de Málaga': 'H.U. Regional de Málaga',
                'Hospital Universitario Virgen de la Victoria': 'H.U. Virgen de la Victoria',
                'Hospital Costa del Sol': 'H. Costa del Sol',
                'Hospital de Antequera': 'H. de Antequera',
                'Hospital de la Axarquía': 'H. de La Axarquía',
                'Hospital Serranía de Ronda': 'H. de la Serranía',
                'Hospital de Benalmádena': 'H. de Benalmádena',
                'Hospital de Alta Resolución Estepona': 'H. de Alta Resolución Estepona',
                'Hospital Universitario Virgen del Rocío': 'H.U. Virgen del Rocío',
                'Hospital Universitario Virgen Macarena': 'H.U. Virgen Macarena',
                'Hospital Virgen de Valme':'H.U. Virgen de Valme',
                'Hospital San Juan de Dios': 'H. San Juan de Dios',
                'Hospital La Merced': 'H. La Merced',
                'Hospital de Écija': 'H. de Écija',
                'Hospital de Lebrija': 'H. de Lebrija',
                'Hospital de Morón de la Frontera': 'H. de Morón de la Frontera',
                'Hospital de Utrera': 'H. de Utrera',
                'Hospital Sierra Norte de Sevilla': 'H. Sierra Norte de Sevilla'}
    
    types_df["Hospital"] = types_df["Hospital"].replace(hospital_mapper)
    types_df = types_df.rename({'Hospital': 'centro'}, axis='columns')

    patients = pd.merge(patients, types_df, on="centro", how='inner')
    patients = patients.rename({'Categoría': 'type_center'}, axis='columns')
    patients["type_center"] = patients["type_center"].astype('category')
    
    print("Recuento tipo de hospitales:")
    print(pd.DataFrame(patients["type_center"].value_counts()))

    return patients