from load_data import load_sgp_data
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model


def build_autoencoder():
    #preprocessing data
    #sgp
    #dataset = load_sgp_data("trigger_1999_2008_summer_hy.nc")
    #features = dataset.iloc[:,0:86]
    #labels = dataset.iloc[:,86]   

	#goamazon
    dataset = load_goamazon_data()
    features = dataset.iloc[:,0:150] 
    labels = dataset.iloc[:,150]

    scaler = StandardScaler()
    scaler.fit(features)
    features_scl = scaler.transform(features)
    
    #build the autoencoder
    input_layer = Input(shape=(150,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(20, activation='relu')(encoded)
    
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(150, activation='relu')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mse')
    autoencoder.fit(features_scl,features_scl,epochs=200,batch_size=256,shuffle=True)
    
    encoder = Model(input_layer, encoded)
    
    features_enc = encoder.predict(features_scl) 
    return features_enc, labels
