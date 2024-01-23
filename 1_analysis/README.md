# Folder Structure

# Dataset description

- `id` (string): Encoded identifier of the patient. It does not have missing values.
- `sex` (binary): Biological sex of the patient.It does not have missing values.
<br>1-> Female
<br>0-> Male
- `center` (category, nominal): "Center where the hospital stay occurs. It does not have missing values.
- `num_shots` (int): Number of Covid vaccine doses administered at the time of admission. A dose is considered administered if at least 14 days have passed. It does not have missing values.
- `vaccinated`: Vaccination status before the admission date.
<br>True-> The patient has been vaccinated
<br>False-> The patient has not been vaccinated
- `icu`(binary): Whether or not the patient has visited the ICU during the current hospital stay.It does not have missing values.
<br> 1 -> has been in the ICU
<br> 0 -> has not been in the ICU
- `province` (category, nominal):  One of the 8 provinces within the autonomous community of Andalusia in Spain where the patient is located or has been treated. There're 8 possible values. It does not have missing values.
- `reinfected` (binary): A binary indicator representing whether the patient has experienced a reinfection. Reinfection is considered if there is a previous admission (with a positive COVID test) with a time difference of at least 50 days compared to another admission time (indicating two separate stays). The computation involves finding the maximum difference between admission dates without considering intermediate stays.It does not have missing values.
<br>1 -> Reinfected
<br>0 -> No reinfected
- `inpatient_days` (int):  The cumulative number of days the patient has been admitted to and spent as an inpatient in the hospital. It does not have missing values.
- `admission_datetime` (date): The date and time when the patient was admitted to the hospital. Just have information about the date. It does not have missing values.
- `discharge_datetime` (date): The date and time when the patient was discharged from the hospital. Just have information about the date. It does not have missing values.
- `hospital_outcome` (binary): The outcome of the hospital stay, indicating the patient's condition upon discharge. It is binary and does not have missing values.
<br>1 -> Deceased
<br>0 -> Alive
- `death_datetime` (date): The date of the patient's death. If the value is missing, it indicates that the patient is alive 30 days after discharge. The information is only the date, not the time. The death date may differ from the discharge date, and it is considered the actual date of the patient's demise if it does not coincide with the discharge date. It has missing values.
- `delta_days_death` (int): The time difference in days between the admission and death of the patient, if applicable. If the patient is still alive, it has a missing value. The information is only about the date, not the time. It has missing values.
- `wave_1` (binary): Encoded with 1 or 0, indicating whether the stay belongs to Wave 1 ("2020-01-01" to "2020-05-10"). It does not have missing values.
<br>1 -> Belongs to wave
<br>0 -> Does not belong to wave
- `wave_2` (binary): Encoded with 1 or 0, indicating whether the stay belongs to Wave 2 ("2020-05-11" to "2020-12-20"). It does not have missing values.
<br>1 -> Belongs to wave
<br>0 -> Does not belong to wave
- `wave_3` (binary): Encoded with 1 or 0, indicating whether the stay belongs to Wave 3 ("2020-12-21" to "2021-03-07"). It does not have missing values.
<br>1 -> Belongs to wave
<br>0 -> Does not belong to wave
- `wave_4` (binary): Encoded with 1 or 0, indicating whether the stay belongs to Wave 4 ("2021-03-08" to "2021-06-20"). It does not have missing values.
<br>1 -> Belongs to wave
<br>0 -> Does not belong to wave
- `wave_5` (binary): Encoded with 1 or 0, indicating whether the stay belongs to Wave 5 ("2021-06-21" to "2021-10-10"). It does not have missing values.
<br>1 -> Belongs to wave
<br>0 -> Does not belong to wave
- `wave_6` (binary): Encoded with 1 or 0, indicating whether the stay belongs to Wave 6 ("2021-10-11" to "2022-03-27"). It does not have missing values.
<br>1 -> Belongs to wave
<br>0 -> Does not belong to wave
- `wave_7` (binary): Encoded with 1 or 0, indicating whether the stay belongs to Wave 7 ("2022-03-28" to "2022-12-31"). It does not have missing values.
<br>1 -> Belongs to wave
<br>0 -> Does not belong to wave
- `pmhx_activecancer` (binary): Encoded with 1 or 0, indicating whether the patient has a history of active cancer. It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_asthma` (binary): Encoded with 1 or 0, indicating whether the patient has a history of asthma. It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_chf` (binary): Encoded with 1 or 0, indicating whether the patient has a history of congestive heart failure (CHF). It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_chronicliver` (binary): Encoded with 1 or 0, indicating whether the patient has a history of chronic liver disease. It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_ckd` (binary): Encoded with 1 or 0, indicating whether the patient has a history of chronic kidney disease (CKD). It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_copd` (binary): Encoded with 1 or 0, indicating whether the patient has a history of chronic obstructive pulmonary disease (COPD). It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_dementia` (binary): Encoded with 1 or 0, indicating whether the patient has a history of dementia. It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_diabetes` (binary): Encoded with 1 or 0, indicating whether the patient has a history of diabetes. It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_hld` (binary): Encoded with 1 or 0, indicating whether the patient has a history of hyperlipidemia (HLD). It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_htn` (binary): Encoded with 1 or 0, indicating whether the patient has a history of hypertension (HTN). It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_ihd` (binary): Encoded with 1 or 0, indicating whether the patient has a history of ischemic heart disease (IHD). It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_obesity` (binary): Encoded with 1 or 0, indicating whether the patient has a history of obesity. It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `pmhx_stroke` (binary): Encoded with 1 or 0, indicating whether the patient has a history of stroke. It does not have missing values.
<br>1-> The patient has the specific comorbidity
<br>0-> The patient does not have the specific comorbidity 
- `lab_alt`(float):: Alanina transaminasa, (U/L). It does not have missing values.
- `lab_ast`(float):  Aspartato transaminasa, (U/L). It does not have missing values.
- `lab_creatinine`(float): Creatinina, (mg/dL). It does not have missing values.
- `lab_crp`(float): Proteína C reactiva, (mg/L). It does not have missing values.
- `lab_ddimer`(float): Dímero-D, (ng/mL). It does not have missing values.
- `lab_glucose`(float): Glucosa, (mg/dL). It does not have missing values.
- `lab_hct`(float): Hematocrito, (%). It does not have missing values.
- `lab_hemoglobin`(float): Hemoglobina, (g/dL). It does not have missing values.
- `lab_inr`(float): Tiempo de protrombina normalizado (INR), (-). It does not have missing values.
- `lab_ldh`(float): Lactato deshidrogenasa, (U/L). It does not have missing values.
- `lab_leukocyte`(float):  Recuento de leucocitos, (x10^3/µL). It does not have missing values.
- `lab_lymphocyte`(float): Recuento de linfocitos, (x10^3/µL). It does not have missing values.
- `lab_lymphocyte_percentage`(float):Porcentaje de linfocitos, (%). It does not have missing values.
- `lab_mch`(float): Hemoglobina corpuscular media, (pg). It does not have missing values.
- `lab_mcv`(float): Volumen corpuscular medio, (fL). It does not have missing values.
- `lab_neutrophil`(float): Recuento de neutrófilos, (x10^3/µL). It does not have missing values.
- `lab_neutrophil_percentage`(float): Porcentaje de neutrófilos, (%). It does not have missing values.
- `lab_platelet`(float): Recuento de plaquetas, (x10^3/µL). It does not have missing values.
- `lab_potassium`(float): Potasio, (mEq/L en HM, mmol/L en otras unidades). It does not have missing values.
- `lab_rbc`(float): Recuento de glóbulos rojos, (x10^6/µL). It does not have missing values.
- `lab_sodium`(float): Sodio, (mEq/L en HM, mmol/L en otras unidades). It does not have missing values.
- `lab_urea`(float): Urea, (mg/dL). It does not have missing values.