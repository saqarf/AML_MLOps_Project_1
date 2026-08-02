import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("tourism_project/data/tourism.csv", index_col=0) # index_col=0 ensures that unnamed: 0 column does not exist
df.drop(columns=['CustomerID'], inplace=True) # Drop unnecessary columns that is not required for our analysis

X=df.drop(columns=["ProdTaken"])
y=df["ProdTaken"]

# Split the data into training and testing sets
#stratify=y keeps the imbalanced failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

print("Data prepared: train/test splits written.")
print("Type values kept as:", sorted(X["TypeofContact"].unique()))
