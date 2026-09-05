import pandas as pd
from src.features.velocity import build_velocity
from src.features.graph import build_graph

def frame():
    return pd.DataFrame({"TransactionDT":[0,60,3600,90000,100000],"TransactionAmt":[10,11,20,30,12],
                         "card1":[1,1,1,2,1],"addr1":[4,4,4,5,4],
                         "P_emaildomain":["a","a","a","b","a"],"DeviceInfo":["d","d","d2","e","d"],
                         "isFraud":[0,1,0,0,1]})

def test_velocity_is_causal():
    x=frame(); full=build_velocity(x); prefix=build_velocity(x.iloc[:3])
    assert full.iloc[:3].reset_index(drop=True).to_numpy().tolist() == prefix.to_numpy().tolist()

def test_graph_is_causal():
    x=frame(); full=build_graph(x); prefix=build_graph(x.iloc[:3])
    assert full.iloc[:3].reset_index(drop=True).to_numpy().tolist() == prefix.to_numpy().tolist()
