import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score
)



DATA="9_results/AAA_risk_stratification_20cases.csv"


df=pd.read_csv(DATA)


print(df)


# label
df["label"] = (
    df["class"]=="AAA"
).astype(int)



# features

X=df[
[
"volume_mm3",
"diameter_mm"
]
]


y=df["label"]



skf=StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



models={

"LogisticRegression":
LogisticRegression(),

"RandomForest":
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

}



results=[]



for name,model in models.items():


    accs=[]
    aucs=[]
    f1s=[]


    for train,test in skf.split(X,y):


        X_train=X.iloc[train]
        X_test=X.iloc[test]


        y_train=y.iloc[train]
        y_test=y.iloc[test]


        scaler=StandardScaler()


        X_train=scaler.fit_transform(
            X_train
        )

        X_test=scaler.transform(
            X_test
        )


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


        accs.append(
            accuracy_score(
                y_test,
                pred
            )
        )


        aucs.append(
            roc_auc_score(
                y_test,
                prob
            )
        )


        f1s.append(
            f1_score(
                y_test,
                pred
            )
        )



    results.append(
        {
        "model":name,
        "Accuracy":np.mean(accs),
        "AUC":np.mean(aucs),
        "F1":np.mean(f1s)
        }
    )



out=pd.DataFrame(results)


out.to_csv(
"9_results/morphology_classifier_results.csv",
index=False
)


print(out)

