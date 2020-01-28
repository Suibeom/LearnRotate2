import LearnR
import numpy as np
import h5py # These are just here for
import uuid # saving and loading models

tsched = np.array([0.3, 0.2, 0.1, 0.0, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0, 0.0, 0.1, 0.2, 0.3, 0.0,
                   0.0, 0.0, 0.1, 0.2, 0.3, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
model = LearnR.model5
id = uuid.uuid1()
model_name = f'model-{id}'
print(model_name)
LearnR.run_sixes_and_nines(model,300)
model.save(model_name+".h5")