dx_dict = {'pmhx_diabetes':['Diabetes'],
                'pmhx_hld':["Dislipemia"],
                'pmhx_htn':["Hipertensión", 
                            "Enfermedad valvular adquirida", 
                            "Arteriopatía de extremidades",
                            ],
                'pmhx_ihd':["Cardiopatía isquémica"],
                'pmhx_ckd':["Insuficiencia renal crónica"],
                'pmhx_copd':["EPOC"],
                'pmhx_asthma':["Asma"],
                'pmhx_activecancer':['Cáncer de mama', 
                                     'Cáncer de próstata', 
                                     "Leucemia",
                                     "Cáncer de bronquio y pulmón",
                                     "Cáncer colorrectal",
                                     "Cáncer de cabeza y cuello"                          
                                     "Cáncer de vejiga",
                                     "Melanoma de piel",
                                     "Cáncer de útero",                                   
                                     "Cáncer de hueso y tejidos blandos",
                                     "Cáncer de testículo",                                
                                     "Cáncer de tiroides",
                                     "Cáncer inmunoproliferativo",
                                     "Cáncer de riñón y pelvis renal",
                                     "Enfermedad de Hodgkin",
                                     "Linfoma no Hodgkin",                                
                                     "Cáncer de páncreas", 
                                     "Cáncer de estómago",
                                     "Cáncer de hígado y vías biliares",
                                     "Cáncer de ovario",
                                     "Cáncer de cuello uterino"
                                     ],
                'pmhx_chronicliver':["Cirrosis hepática", "Esteatosis hepática"],
                'pmhx_stroke':["Secuela enfermedad cerebrovascular",
                               "Enfermedad cerebrovascular mal definida y otra"],
                'pmhx_chf':['Insuficiencia cardiaca'],
                'pmhx_dementia':['Demencia', 
                                 "Otro trastorno mental orgánico", 
                                 "Trastorno esquizofrénico"],
                'pmhx_obesity': ["Obesidad"],
                'pmhx_tobacco': ['Dependencia tabaco']}

lab_map = {"Creatinina": "lab_creatinine",                        # unidad mg/dl    (503)                  
                "Glucosa": "lab_glucose",                               # unidad mg/dl    (500)
                "Aspartato transaminasa": "lab_ast",                    # unidad U/L      (541)
                "Alanina transaminasa": "lab_alt",                      # unidad U/L      (542)
                "Potasio": "lab_potassium",                             # unidad mEq/L    (508), en HM unidad mmol/L                     
                "Sodio": "lab_sodium",                                  # unidad mEq/L    (506), en HM unidad mmol/L
                "Hematocrito": "lab_hct",                               # unidad %        (193)
                "Hemoglobina": "lab_hemoglobin",                        # unidad g/DL     (195)
                "Linfocitos (porcentaje)": "lab_lymphocyte_percentage", # unidad %        (175)
                "Linfocitos (recuento)": "lab_lymphocyte",              # unidad x10^3/µL (198) 
                "Hemoglobina corpuscular media": "lab_mch",             # unidad pg       (197)
                "Volumen corpuscular medio": "lab_mcv",                 # unidad fL       (194)
                "Neutrófilos (porcentaje)": "lab_neutrophil_percentage",# unidad %                     
                "Neutrófilos (recuento)": "lab_neutrophil",             # unidad x10^3/µL (174)
                "Plaquetas (recuento)": "lab_platelet",                 # unidad x10^3/µL (198)
                "Hematíes (recuento)": "lab_rbc",                       # unidad x10^6/µL (192)
                "Leucocitos (recuento)": "lab_leukocyte",               # unidad x10^3/µL (173)
                "Urea": "lab_urea",                                     # unidad mg/dl    (503)                         
                "Lactato deshidrogenasa": "lab_ldh",                    # unidad U/L      (525)
                "Proteína C reactiva": "lab_crp",                       # unidad mg/L     (636)   
                "Tiempo de protrombina normalizado (INR)": "lab_inr",   # unidad -        (164)
                "Dímero-D": "lab_ddimer",                               # unidad ng/mL    (115)                
                "Tiempo de protrombina": "lab_prothrombin",             # unidad NA       (266)  - equivalencia con HM desconocida 
    }

hospital_province_map = {"H. de Baza": "Granada",                      
                            "H. de La Axarquía": "Málaga",                       
                            "H.U. Torrecárdenas": "Almería",                       
                            "H.U. Juan Ramón Jiménez": "Huelva",                
                            "H. San Agustín": "Jaén",                        
                            "H. Costa del Sol": "Málaga",                       
                            "H.U. Virgen del Rocío": "Sevilla",                 
                            "H.U. Virgen de la Victoria": "Málaga",             
                            "H.U. Reina Sofía": "Córdoba",                       
                            "H. Infanta Elena": "Huelva",                       
                            "H.U. de Jaén": "Jaén",                           
                            "H. Santa Ana": "Granada",                          
                            "H.U. Virgen de las Nieves": "Granada",              
                            "H. San Juan de Dios del Aljarafe": "Sevilla",      
                            "H.U. Regional de Málaga": "Málaga",                
                            "H.U. San Cecilio": "Granada",                       
                            "H.U. Puerta del Mar": "Cádiz",                    
                            "H. Infanta Margarita": "Córdoba",                  
                            "H.U. de Puerto Real": "Cádiz",                   
                            "H. Punta de Europa": "Cádiz",                     
                            "H.U. de Jerez de la Frontera": "Cádiz",           
                            "H. de Poniente": "Almería",                        
                            "H. La Inmaculada": "Almería",                      
                            "H.U. Virgen Macarena": "Sevilla",                   
                            "H. Valle de los Pedroches": "Córdoba",             
                            "H. de La Línea de la Concepción": "Cádiz",        
                            "H. San Juan de la Cruz ": "Jaén",                
                            "H. de la Serranía": "Málaga",                     
                            "H.U. Virgen de Valme": "Sevilla",                  
                            "H. de Antequera": "Málaga",                       
                            "H. La Merced": "Sevilla",                          
                            "H. de Riotinto": "Huelva",                        
                            "H. Alto Guadalquivir": "Jaén",                  
                            "Hospital General Santa María del Puerto": "Cádiz",
                            "Hospital Virgen Bella": "Huelva",                  
                            "Hospital San Rafael": "Granada",                  
                            "Hospital Virgen de Las Montañas": "Cádiz",   
                            "H. San Juan de la Cruz": "Jaén",
    }