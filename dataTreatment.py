import pandas as pd
import matplotlib.pyplot as plt

def score_ph(val):
    if 6.5 <= val <= 8.5:
        return 100
    elif 6 <= val <= 9:
        return 70
    else:
        return 0

def score_no3(val):
    if val < 25:
        return 100
    elif val <= 50:
        return 60
    else:
        return 0

def score_turb(val):
    if val < 1:
        return 100
    elif val <= 5:
        return 70
    elif val <= 25:
        return 40
    else:
        return 0

def score_o2(val):
    if val > 80:
        return 100
    elif val >= 60:
        return 70
    elif val >= 40:
        return 30
    else:
        return 0

def score_phyco(val):
    if val < 10:
        return 100
    elif val <= 20:
        return 50
    else:
        return 0

def compute_score(row):
    # règles bloquantes
    if row["Phycocyanine scaled"] > 20 or row["Turbidité"] > 50:
        return 0

    score = (
        score_phyco(row["Phycocyanine scaled"]) * 0.25 +
        score_turb(row["Turbidité"]) * 0.15 +
        score_o2(row["O2 Saturation"]) * 0.15 +
        score_ph(row["pH Test"]) * 0.10 +
        score_no3(row["NO3"]) * 0.05
    )

    return round(score, 1)


file = pd.read_csv("Consibio Cloud Datalog.csv")
file = file.dropna(subset=["O2 Saturation","Turbidité","NO3","Phycocyanine scaled","pH Test"])
file["Date"] = pd.to_datetime(file["Date"],dayfirst=True,format="mixed")
file["Date"] = file["Date"].dt.strftime("%d/%m-%y")
file=file.groupby("Date").mean()
file["Score_Baignade"]=file.apply(compute_score,axis=1)
if True:
    ax = plt.subplot()
    ax.plot(file["Score_Baignade"])
    plt.xticks(rotation=90)
    plt.show()