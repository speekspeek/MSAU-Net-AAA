import pandas as pd
import numpy as np


from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline


from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC


from xgboost import XGBClassifier


from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)



DATA = (
"9_results/AAA_risk_stratification_20cases.csv"
)



df=pd.read_csv(DATA)



print(df)



df["label"] = (
    df["class"]=="AAA"
).astype(int)



# =========================
# Morphological features
# =========================

features=[

    "volume_mm3",

    "diameter_mm",

    "surface_area_mm2",

    "sphericity",

    "volume_diameter_ratio"

]


# 如果旧文件没有后三个指标
# 自动兼容

features=[
    f for f in features
    if f in df.columns
]


print(
"Features:",
features
)



X=df[features]

y=df["label"]



models={


"LogisticRegression":

Pipeline(
[
("scaler",
 StandardScaler()),

("model",
 LogisticRegression(
     max_iter=1000
 ))
]
),



"RandomForest":

RandomForestClassifier(
    n_estimators=300,
    random_state=42
),



"SVM":

Pipeline(
[
("scaler",
 StandardScaler()),

("model",
 SVC(
     probability=True,
     kernel="rbf"
 ))
]
),



"XGBoost":

XGBClassifier(

    n_estimators=200,

    max_depth=3,

    learning_rate=0.05,

    subsample=0.8,

    random_state=42,

    eval_metric="logloss"

)

}




skf=StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)



results=[]



for name,model in models.items():


    print("\nModel:",name)


    acc=[]
    auc=[]
    f1=[]
    prec=[]
    rec=[]



    for train_idx,test_idx in skf.split(X,y):


        X_train=X.iloc[train_idx]

        X_test=X.iloc[test_idx]


        y_train=y.iloc[train_idx]

        y_test=y.iloc[test_idx]



        model.fit(
            X_train,
            y_train
        )



        pred=model.predict(
            X_test
        )



        prob=model.predict_proba(
            X_test
        )[:,1]



        acc.append(
            accuracy_score(
                y_test,
                pred
            )
        )


        auc.append(
            roc_auc_score(
                y_test,
                prob
            )
        )


        f1.append(
            f1_score(
                y_test,
                pred
            )
        )


        prec.append(
            precision_score(
                y_test,
                pred,
                zero_division=0
            )
        )


        rec.append(
            recall_score(
                y_test,
                pred
            )
        )



    results.append(

    {

    "Model":name,

    "Accuracy":np.mean(acc),

    "AUC":np.mean(auc),

    "F1":np.mean(f1),

    "Precision":np.mean(prec),

    "Recall":np.mean(rec)

    }

    )



out=pd.DataFrame(results)



out.to_csv(

"9_results/morphology_classifier_v2_results.csv",

index=False

)



print("\n===== RESULTS =====")

print(out)


