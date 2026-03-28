import sys
sys.path.append('..')  # '..' = back to the previous folder (here = NCTE)

import prepare_dataset

vars_class = [
    "CLPC", "CLNC", "CLTS", "CLRSP", "CLBM", "CLPRDT", "CLILF", 
    "CLCU", "CLAPS", "CLQF", "CLINSTD", "CLSTENG"
]
vars_mqi = [
    "DIRINST", "WCDISS", "APLPROB", "CWCM", "LINK", "EXPL", "MMETH", "MGEN", "MLANG",
    "ORICH", "ORICH4", "REMED", "USEPROD", "MATCON", "OWWS", "OWWS4", "MAJERR",
    "LANGIMP", "LCP", "OERR", "OERR4", "STEXPL", "SMQR", "ETCA", "OSPMMR", "OSPMMR4", 
    "STUCON", "STUCOM", "SMALDIS", "MMSM", "ORIENT", "SUMM", "MQI_CHECK", "DIFFINST",  
    "LLC", "MQI3", "MKT3", "MQI5", "MKT5", "TSTUDEA", "TREMSTU", "STUENG", "CLMATINQ",  
    "LESSEFFIC", "DENSMAT", "LATASK", "LESSCLEAR", "TASKDEVMAT", "ERRANN", "WORLD"
]

for var in vars_class:
    df = prepare_dataset.prepare_dataset(var, MQI=False)
    file_title = "CLASS_" + var + "_dataset.csv"
    df.to_csv(file_title, index=False, encoding="utf-8")

for var in vars_mqi:
    df = prepare_dataset.prepare_dataset(var, MQI=True)
    file_title = "MQI_" + var + "_dataset.csv"
    df.to_csv(file_title, index=False, encoding="utf-8")