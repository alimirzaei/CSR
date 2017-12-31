from flask import Flask
from flask import request
import json
import os
import numpy as np
from ChannelEstimatorCNN import ChannelEstimatorCNN
app = Flask(__name__)

model = ChannelEstimatorCNN()
model.loadModel('cnn')

@app.route("/estimate_channel_2D", methods = ['POST'])
def estimate_channel_2D():
#	global model
	data = json.loads(request.data.decode('utf-8'))

	image=np.array(data['image'])
	image = np.moveaxis(image, 0, -1)
	Noise_var = data['Noise_var']

	image=image.reshape(1,image.shape[0],image.shape[1], 2)
	print(image.shape)
	y = model.test(image)
	print(y.shape)
	result = json.dumps(y[0].tolist())
	return result

if __name__ == "__main__":
	app.run()	